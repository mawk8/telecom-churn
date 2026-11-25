"""Build point-in-time subscriber features from the synthetic TDG delivery."""

import csv
import io
import json
import logging
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path
from tempfile import gettempdir
from uuid import uuid4
from zipfile import ZipFile

import duckdb
from fsspec.implementations.zip import ZipFileSystem


LOGGER = logging.getLogger(__name__)
PASSPORT = "reference/passport.json"
TRUTH = "reference/churn_truth.csv"
MANIFEST = "MANIFEST.csv"
SOURCES = {"cdr": "cdr.csv", "tickets": "tickets.csv", "telemetry": "telemetry.jsonl"}


@contextmanager
def _source(path):
    """Present the directory and archive as the same read-only tree."""
    if isinstance(path, Path) and path.is_dir():
        yield (
            lambda name: (path / name).open("rb"),
            lambda name: (path / name).stat().st_size,
            lambda name: (path / name).as_posix(),
            None,
        )
        return

    with ZipFile(path) as archive:
        prefix = "out/" if f"out/{PASSPORT}" in archive.namelist() else ""

        def size(name):
            try:
                return archive.getinfo(prefix + name).file_size
            except KeyError as error:
                raise FileNotFoundError(prefix + name) from error

        def open_member(name):
            size(name)
            return archive.open(prefix + name)

        yield open_member, size, lambda name: f"zip://{prefix}{name}", path


def load_data(input_path, include_late=False):
    """Validate the expected partitions and return a descriptor consumed by preprocess."""
    path = input_path if hasattr(input_path, "read") else Path(input_path).resolve()
    with _source(path) as (open_file, size, _, _):
        with open_file(PASSPORT) as file:
            passport = json.load(file)
        with open_file(MANIFEST) as file:
            manifest = {
                row["path"]: int(row["bytes"])
                for row in csv.DictReader(io.TextIOWrapper(file, encoding="utf-8-sig"))
            }
        start = date.fromisoformat(passport["period_start"])
        missing = set(passport["missing_day_indices"])
        broken = passport["broken_day_index"]
        partitions = {source: [] for source in SOURCES}
        skipped_damaged = []
        for day in range(passport["period_days"]):
            dt = start + timedelta(days=day)
            layer = "late_delivery" if day in missing else "raw"
            for source, filename in SOURCES.items():
                name = f"{layer}/{source}/dt={dt}/{filename}"
                if source == "cdr" and day == broken:
                    skipped_damaged.append(name)
                    continue
                if layer == "raw" or include_late:
                    partitions[source].append((day, name))
        for name in [TRUTH, *(name for rows in partitions.values() for _, name in rows)]:
            actual_bytes = size(name)
            if name not in manifest:
                raise ValueError(f"Отсутствует в MANIFEST.csv: {name}")
            if actual_bytes != manifest[name]:
                raise ValueError(f"Размер файла отличается от MANIFEST.csv: {name}")

    quality = {
        "include_late": bool(include_late),
        "known_missing_days": [str(start + timedelta(days=d)) for d in sorted(missing)],
        "skipped_damaged": skipped_damaged,
        "selected_partitions": {source: len(rows) for source, rows in partitions.items()},
    }
    return {"path": path, "passport": passport, "partitions": partitions, "quality": quality}


def input_files(input_path, include_late=False):
    """Return the local files that determine a run fingerprint."""
    path = Path(input_path).resolve()
    if not path.is_dir():
        return (path,)
    data = load_data(path, include_late)
    names = [PASSPORT, MANIFEST, TRUTH]
    names.extend(name for rows in data["partitions"].values() for _, name in rows)
    return tuple(path / name for name in sorted(names))


def _read_sources(db, data, locate, max_day):
    start = date.fromisoformat(data["passport"]["period_start"])
    end = start + timedelta(days=max_day)
    paths = {
        source: [locate(name) for day, name in rows if day < max_day]
        for source, rows in data["partitions"].items()
    }
    quality = dict(data["quality"], read_partitions={k: len(v) for k, v in paths.items()})
    db.execute("""
        CREATE TEMP TABLE truth AS
        SELECT subscriber_key, CAST(churn_day_index AS INTEGER) AS churn_day
        FROM read_csv(?, header = true, all_varchar = true)
    """, [locate(TRUTH)])
    count, distinct_count, bad = db.execute("""
        SELECT count(*), count(DISTINCT subscriber_key),
               count(*) FILTER (WHERE churn_day IS NULL OR churn_day < -1 OR churn_day >= ?)
        FROM truth
    """, [data["passport"]["period_days"]]).fetchone()
    if count != data["passport"]["subscribers"] or count != distinct_count or bad:
        raise ValueError("churn_truth.csv: неверное число, ключи или даты абонентов")

    db.execute("""
        CREATE TEMP TABLE cdr AS
        SELECT record_id::BIGINT AS record_id, subscriber_key, node_id,
               coalesce(try_cast(event_ts AS TIMESTAMP),
                        strptime(event_ts, '%d.%m.%Y %H:%M:%S')) AS event_ts,
               ingest_ts::TIMESTAMP AS ingest_ts,
               CASE WHEN bytes_up::BIGINT BETWEEN 0 AND 499999999999
                     AND bytes_down::BIGINT BETWEEN 0 AND 499999999999
                    THEN bytes_up::BIGINT + bytes_down::BIGINT END AS bytes,
               roaming_flag::BOOLEAN AS roaming
        FROM read_csv(?, header = true, all_varchar = true)
    """, [paths["cdr"]])
    db.execute("""
        CREATE TEMP TABLE tickets AS
        SELECT ticket_id::BIGINT AS ticket_id, subscriber_key,
               created_ts::TIMESTAMP AS created_ts
        FROM read_csv(?, header = true, all_varchar = true)
    """, [paths["tickets"]])
    db.execute("""
        CREATE TEMP TABLE telemetry AS
        SELECT ts, node_id, load_pct, latency_ms, error_rate
        FROM read_json(?, format = 'newline_delimited',
            columns = {
                ts: 'TIMESTAMP', node_id: 'VARCHAR', load_pct: 'DECIMAL(24,9)',
                latency_ms: 'DECIMAL(24,9)', error_rate: 'DECIMAL(24,9)'
            })
    """, [paths["telemetry"]])
    quality["source_rows"] = {
        source: db.execute(f"SELECT count(*) FROM {source}").fetchone()[0]
        for source in ("truth", *SOURCES)
    }
    invalid_cdr, invalid_bytes = db.execute("""
        SELECT count(*) FILTER (WHERE subscriber_key IS NULL),
               count(*) FILTER (WHERE bytes IS NULL)
        FROM cdr
    """).fetchone()
    quality["cdr_rows_without_subscriber"] = invalid_cdr
    quality["cdr_rows_without_valid_bytes"] = invalid_bytes
    for source, criteria in (
        ("cdr", "record_id IS NULL OR event_ts IS NULL OR ingest_ts IS NULL"),
        ("tickets", "ticket_id IS NULL OR subscriber_key IS NULL OR created_ts IS NULL"),
        ("telemetry", "ts IS NULL OR node_id IS NULL"),
    ):
        if db.execute(f"SELECT EXISTS (SELECT 1 FROM {source} WHERE {criteria})").fetchone()[0]:
            raise ValueError(f"{source}: отсутствуют обязательные ключи или время")
    LOGGER.info("Строки входных источников до %s: %s", end, quality["source_rows"])
    return quality


def _cutoff_frame(db, start, day, horizon, window):
    end = start + timedelta(days=day)
    mid = end - timedelta(days=window)
    low = mid - timedelta(days=window)
    return db.execute("""
        WITH params AS (
            SELECT ?::DATE AS low, ?::DATE AS mid, ?::DATE AS cutoff,
                   ?::INTEGER AS day, ?::INTEGER AS horizon
        ),
        eligible AS (
            SELECT subscriber_key, (churn_day >= day AND churn_day < day + horizon)::INTEGER AS label
            FROM truth, params WHERE churn_day = -1 OR churn_day >= day
        ),
        available_cdr AS MATERIALIZED (
            SELECT cdr.* FROM cdr, params
            WHERE event_ts < cutoff AND ingest_ts < cutoff AND subscriber_key IS NOT NULL
            QUALIFY row_number() OVER (
                PARTITION BY record_id
                ORDER BY ingest_ts, event_ts, subscriber_key, node_id, bytes, roaming
            ) = 1
        ),
        activity AS (
            SELECT subscriber_key,
                   count(*) FILTER (WHERE event_ts >= mid) AS cdr_events_recent,
                   count(*) FILTER (WHERE event_ts >= low AND event_ts < mid) AS cdr_events_previous,
                   count(DISTINCT event_ts::DATE) FILTER (WHERE event_ts >= mid) AS cdr_active_days_recent,
                   count(DISTINCT event_ts::DATE) FILTER (WHERE event_ts >= low AND event_ts < mid) AS cdr_active_days_previous,
                   coalesce(sum(bytes) FILTER (WHERE event_ts >= mid), 0)::DOUBLE AS cdr_bytes_recent,
                   coalesce(sum(bytes) FILTER (WHERE event_ts >= low AND event_ts < mid), 0)::DOUBLE AS cdr_bytes_previous,
                   count(*) FILTER (WHERE event_ts >= mid AND roaming) AS cdr_roaming_recent,
                   count(*) FILTER (WHERE event_ts >= low AND event_ts < mid AND roaming) AS cdr_roaming_previous,
                   date_diff('day', max(event_ts)::DATE, cutoff) AS cdr_recency_days
            FROM available_cdr, params GROUP BY subscriber_key, cutoff
        ),
        node_counts AS (
            SELECT subscriber_key, node_id, count(*) AS events
            FROM available_cdr, params
            WHERE event_ts >= low AND node_id IS NOT NULL
            GROUP BY subscriber_key, node_id
        ),
        primary_node AS (
            SELECT subscriber_key, node_id FROM node_counts
            QUALIFY row_number() OVER (PARTITION BY subscriber_key ORDER BY events DESC, node_id) = 1
        ),
        node_metrics AS (
            SELECT node_id, count(*) AS tel_samples,
                   avg(load_pct) AS tel_load_pct_mean, avg(latency_ms) AS tel_latency_ms_mean,
                   avg(error_rate) AS tel_error_rate_mean
            FROM telemetry, params WHERE ts >= low AND ts < cutoff GROUP BY node_id
        ),
        -- RC9 twins differ from the original in closure fields only.
        distinct_tickets AS (
            SELECT ticket_id, subscriber_key, created_ts
            FROM tickets, params WHERE created_ts < cutoff
            QUALIFY row_number() OVER (PARTITION BY ticket_id ORDER BY created_ts, subscriber_key) = 1
        ),
        ticket_counts AS (
            SELECT subscriber_key,
                   count(*) FILTER (WHERE created_ts >= mid) AS ticket_count_recent,
                   count(*) FILTER (WHERE created_ts < mid) AS ticket_count_previous
            FROM distinct_tickets, params WHERE created_ts >= low GROUP BY subscriber_key
        )
        SELECT eligible.subscriber_key, eligible.label,
               coalesce(cdr_events_recent, 0) AS cdr_events_recent,
               coalesce(cdr_events_previous, 0) AS cdr_events_previous,
               coalesce(cdr_active_days_recent, 0) AS cdr_active_days_recent,
               coalesce(cdr_active_days_previous, 0) AS cdr_active_days_previous,
               coalesce(cdr_bytes_recent, 0) AS cdr_bytes_recent,
               coalesce(cdr_bytes_previous, 0) AS cdr_bytes_previous,
               coalesce(cdr_roaming_recent, 0) AS cdr_roaming_recent,
               coalesce(cdr_roaming_previous, 0) AS cdr_roaming_previous,
               coalesce(cdr_events_recent, 0) - coalesce(cdr_events_previous, 0) AS cdr_events_delta,
               coalesce(cdr_active_days_recent, 0) - coalesce(cdr_active_days_previous, 0) AS cdr_active_days_delta,
               coalesce(cdr_bytes_recent, 0) - coalesce(cdr_bytes_previous, 0) AS cdr_bytes_delta,
               coalesce(cdr_roaming_recent / nullif(cdr_events_recent, 0), 0) AS cdr_roaming_share_recent,
               coalesce(cdr_roaming_previous / nullif(cdr_events_previous, 0), 0) AS cdr_roaming_share_previous,
               cdr_recency_days::DOUBLE AS cdr_recency_days,
               (cdr_recency_days IS NOT NULL)::INTEGER AS cdr_has_history,
               coalesce(ticket_count_recent, 0) AS ticket_count_recent,
               coalesce(ticket_count_previous, 0) AS ticket_count_previous,
               coalesce(tel_samples, 0) AS tel_samples,
               tel_load_pct_mean, tel_latency_ms_mean, tel_error_rate_mean
        FROM eligible CROSS JOIN params
        LEFT JOIN activity USING (subscriber_key)
        LEFT JOIN ticket_counts USING (subscriber_key)
        LEFT JOIN primary_node USING (subscriber_key)
        LEFT JOIN node_metrics USING (node_id)
        ORDER BY eligible.subscriber_key
    """, [low, mid, end, day, horizon]).df()


def _cutoff_source_rows(db, cutoff):
    counts = {}
    for source, key, condition, dates in (
        ("cdr", "record_id", "event_ts < ? AND ingest_ts < ? AND subscriber_key IS NOT NULL",
         [cutoff, cutoff]),
        ("tickets", "ticket_id", "created_ts < ?", [cutoff]),
    ):
        candidates, unique = db.execute(f"""
            SELECT count(*), count(DISTINCT {key}) FROM {source} WHERE {condition}
        """, dates).fetchone()
        counts[source] = {"candidates": candidates, "unique": unique}
    LOGGER.info("Доступные записи до %s и уникальные ключи: %s", cutoff, counts)
    return counts


def preprocess(data, cfg):
    """Build chronological training and holdout frames with one row per eligible subscriber."""
    train_day = cfg["train_day"]
    test_day = cfg["test_day"]
    horizon = cfg["horizon_days"]
    window = cfg["window_days"]
    passport = data["passport"]
    if any(type(value) is not int for value in (train_day, test_day, horizon, window)):
        raise ValueError("Параметры дней должны быть целыми числами")
    if not (0 < window and 0 < horizon and 2 * window <= train_day
            and train_day + horizon <= test_day and test_day + horizon <= passport["period_days"]):
        raise ValueError("Окна и горизонты должны помещаться в период без пересечения горизонтов")
    start = date.fromisoformat(passport["period_start"])
    skipped = data["quality"]["skipped_damaged"]
    if skipped:
        LOGGER.warning("Повреждённая партиция CDR пропущена: %s", ", ".join(skipped))
    cutoffs = {"train": start + timedelta(days=train_day),
               "test": start + timedelta(days=test_day)}
    # DuckDB owns this isolated spill path and removes its spill data on close.
    spill = Path(gettempdir()) / f"churn-duckdb-{uuid4().hex}"
    with _source(data["path"]) as (_, _, locate, archive):
        with duckdb.connect(config={"threads": 2, "memory_limit": "1GB",
                                    "temp_directory": str(spill), "max_temp_directory_size": "4GB",
                                    "preserve_insertion_order": False}) as db:
            db.execute("SET enable_progress_bar = false")
            if archive is not None:
                filesystem = ZipFileSystem(str(archive) if isinstance(archive, Path) else archive,
                                           skip_instance_cache=True)
                try:
                    db.register_filesystem(filesystem)
                    quality = _read_sources(db, data, locate, test_day)
                finally:
                    filesystem.close()
            else:
                quality = _read_sources(db, data, locate, test_day)
            train = _cutoff_frame(db, start, train_day, horizon, window)
            test = _cutoff_frame(db, start, test_day, horizon, window)
            quality["cutoff_source_rows"] = {
                split: _cutoff_source_rows(db, cutoff) for split, cutoff in cutoffs.items()
            }
    quality["cutoff_dates"] = {split: str(cutoff) for split, cutoff in cutoffs.items()}
    quality["cohort_rows"] = {"train": len(train), "test": len(test)}
    return train, test, quality

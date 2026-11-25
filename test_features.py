import json
import unittest
from datetime import date, timedelta
from io import BytesIO
from zipfile import ZipFile

from features import load_data, preprocess
from pandas import isna


def sample_zip(omit=None):
    start = date(2025, 10, 1)
    cdr_header = "record_id,event_ts,ingest_ts,subscriber_key,bytes_up,bytes_down,node_id,roaming_flag\n"
    ticket_header = "ticket_id,created_ts,subscriber_key,resolution_code\n"
    files = {
        "reference/passport.json": json.dumps({
            "period_start": str(start), "period_days": 6,
            "missing_day_indices": [2], "broken_day_index": 5, "subscribers": 3,
        }),
        "reference/churn_truth.csv": "subscriber_key,churn_day_index\na,2\nb,-1\nc,4\n",
    }
    for day in range(6):
        prefix = "late_delivery" if day == 2 else "raw"
        dt = start + timedelta(days=day)
        cdr = f"{prefix}/cdr/dt={dt}/cdr.csv"
        ticket = f"{prefix}/tickets/dt={dt}/tickets.csv"
        telemetry = f"{prefix}/telemetry/dt={dt}/telemetry.jsonl"
        row = f"{day},{dt} 10:00:00,{dt} 13:00:00,b,1,2,n1,true\n"
        files[cdr] = cdr_header + row if day != 5 else "damaged CDR export\n"
        if day == 1:
            files[cdr] += (
                row
                + f"10,02.10.2025 10:00:00,2025-10-03 00:00:00,b,1,2,n1,false\n"
            )
        files[ticket] = (
            ticket_header + f"{day},{dt} 13:00:00,b,RC1\n"
            + f"{day},{dt} 13:00:00,b,RC9\n"
        )
        files[telemetry] = json.dumps({
            "ts": f"{dt}T10:00:00", "node_id": "n1", "load_pct": 17 if day < 2 else 99,
            "latency_ms": 25, "error_rate": 0.01,
        }) + "\n"
    files["MANIFEST.csv"] = "path,bytes\n" + "".join(
        f"{path},{len(body.encode('utf-8'))}\n" for path, body in files.items()
    )
    archive = BytesIO()
    with ZipFile(archive, "w") as output:
        for name, body in files.items():
            if name != omit:
                output.writestr(f"out/{name}", body)
    archive.seek(0)
    return archive


class FeatureTests(unittest.TestCase):
    cfg = {"train_day": 2, "test_day": 4, "horizon_days": 2, "window_days": 1}

    def test_cutoffs_labels_dedup_and_late_delivery(self):
        train, test, quality = preprocess(load_data(sample_zip()), self.cfg)
        train = train.set_index("subscriber_key")
        test = test.set_index("subscriber_key")
        self.assertEqual(train["label"].to_dict(), {"a": 1, "b": 0, "c": 0})
        self.assertEqual(test["label"].to_dict(), {"b": 0, "c": 1})
        self.assertEqual(train.loc["b", "cdr_events_recent"], 1)
        self.assertEqual(train.loc["b", "cdr_bytes_recent"], 3)
        self.assertEqual(train.loc["b", "cdr_roaming_share_recent"], 1)
        self.assertEqual(train.loc["b", "cdr_events_delta"], 0)
        self.assertEqual(train.loc["a", "cdr_has_history"], 0)
        self.assertTrue(isna(train.loc["a", "cdr_recency_days"]))
        self.assertEqual(train.loc["b", "ticket_count_recent"], 1)
        self.assertEqual(train.loc["b", "tel_load_pct_mean"], 17)
        self.assertEqual(test.loc["b", "cdr_events_previous"], 0)
        self.assertEqual(quality["known_missing_days"], ["2025-10-03"])
        self.assertEqual(quality["cutoff_source_rows"]["train"]["cdr"], {
            "candidates": 3, "unique": 2,
        })

        _, test_late, _ = preprocess(load_data(sample_zip(), include_late=True), self.cfg)
        self.assertEqual(test_late.set_index("subscriber_key").loc["b", "cdr_events_previous"], 1)

    def test_missing_raw_partition_raises(self):
        with self.assertRaisesRegex(FileNotFoundError, "2025-10-02"):
            load_data(sample_zip("raw/cdr/dt=2025-10-02/cdr.csv"))

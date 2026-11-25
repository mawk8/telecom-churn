"""Build and publish a reproducible churn model from the generated sources."""

import argparse
import hashlib
import json
import logging
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PROJECT = Path(__file__).resolve().parent
ARTIFACTS = (
    "train.parquet",
    "test.parquet",
    "model.joblib",
    "metrics.json",
    "data_quality.json",
    "pipeline.log",
)
MARKER = "_SUCCESS.json"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
logger = logging.getLogger(__name__)


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def file_digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def fingerprint(input_path, paths, config, include_late):
    sources = {
        path.relative_to(input_path).as_posix() if input_path.is_dir() else path.name:
        file_digest(path)
        for path in paths
    }
    # Hash the actual data bytes. The input manifest describes delivered files,
    # but its contents alone cannot detect overwritten partition files.
    payload = {"config": config, "include_late": include_late, "sources": sources}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def runtime():
    try:
        xgboost_version = version("xgboost")
        xgboost_package = "xgboost"
    except PackageNotFoundError:
        xgboost_package = "xgboost-cpu"
        xgboost_version = version(xgboost_package)
    packages = (
        "duckdb", "fsspec", "numpy", "pandas", "pyarrow", "scikit-learn",
        "scipy", "joblib", "threadpoolctl",
    )
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "platform": sys.platform,
        "packages": {**{name: version(name) for name in packages}, xgboost_package: xgboost_version},
    }


def completed(directory):
    try:
        with (directory / MARKER).open(encoding="utf-8") as marker:
            manifest = json.load(marker)
        if not isinstance(manifest, dict) or manifest.get("run_id") != directory.name:
            return False
        hashes = manifest.get("artifact_sha256")
        if not isinstance(hashes, dict) or hashes.keys() != set(ARTIFACTS):
            return False
        return all(file_digest(directory / name) == hashes[name] for name in ARTIFACTS)
    except (OSError, ValueError):
        return False


def build_artifacts(staging, input_path, config, include_late):
    import joblib
    import numpy as np

    from features import load_data, preprocess
    from modeling import evaluate, train_model

    log_handler = logging.FileHandler(staging / "pipeline.log", encoding="utf-8")
    log_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger = logging.getLogger()
    root_logger.addHandler(log_handler)
    try:
        logger.info("Loading input from %s (late deliveries: %s)", input_path, include_late)
        data = load_data(input_path, include_late=include_late)
        logger.info("Building train and test snapshots")
        train_frame, test_frame, quality = preprocess(data, config)
        logger.info("Training model on %d rows", len(train_frame))
        bundle, validation = train_model(train_frame, config)
        test = evaluate(bundle, test_frame)
        logger.info("Saving snapshots and checking model serialization")
        train_frame.to_parquet(staging / "train.parquet", index=False)
        test_frame.to_parquet(staging / "test.parquet", index=False)
        model_path = staging / "model.joblib"
        joblib.dump(bundle, model_path)
        restored = joblib.load(model_path)
        np.testing.assert_array_equal(
            bundle["estimator"].predict_proba(test_frame[bundle["features"]]),
            restored["estimator"].predict_proba(test_frame[restored["features"]]),
        )
        if restored["threshold"] != bundle["threshold"]:
            raise RuntimeError("Model threshold changed after serialization")
        write_json(staging / "metrics.json", {"validation": validation, "test": test})
        write_json(staging / "data_quality.json", quality)
        logger.info("Saved artifacts and verified predictions on %d rows", len(test_frame))
    except Exception:
        logger.exception("Pipeline failed. Partial artifacts are in %s", staging)
        raise
    finally:
        root_logger.removeHandler(log_handler)
        log_handler.close()


def run_pipeline(input_path, output_path, config, include_late=False):
    from features import input_files

    sources = input_files(input_path, include_late=include_late)
    runtime_info = runtime()
    code = {
        name: file_digest(PROJECT / name)
        for name in ("run_pipeline.py", "features.py", "modeling.py", "requirements.txt")
    }
    identity = {"settings": config, "code": code, "runtime": runtime_info}
    run_id = fingerprint(input_path, sources, identity, include_late)
    destination = output_path / run_id
    if completed(destination):
        logger.info("Using completed run %s", destination)
        return destination
    if destination.exists():
        raise FileExistsError(f"Incomplete or corrupt destination already exists: {destination}")

    output_path.mkdir(exist_ok=True, parents=True)
    staging = Path(tempfile.mkdtemp(prefix=f".pending-{run_id}-", dir=output_path))
    build_artifacts(staging, input_path, config, include_late)
    missing = [artifact for artifact in ARTIFACTS if not (staging / artifact).is_file()]
    if missing:
        raise RuntimeError(f"Missing artifacts in {staging}: {', '.join(missing)}")
    current_sources = input_files(input_path, include_late=include_late)
    if fingerprint(input_path, current_sources, identity, include_late) != run_id:
        raise RuntimeError(f"Input changed during execution. Partial artifacts are in {staging}")
    write_json(staging / MARKER, {
        "run_id": run_id,
        "config": config,
        "include_late": include_late,
        "runtime": runtime_info,
        "artifacts": list(ARTIFACTS),
        "artifact_sha256": {name: file_digest(staging / name) for name in ARTIFACTS},
    })
    try:
        staging.rename(destination)
    except OSError:
        # Another process can publish the same run while we prepare it.
        if not completed(destination):
            raise
        logger.info("Another process published %s. Staging remains in %s", destination, staging)
    else:
        logger.info("Published run %s", destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT / "config.json")
    parser.add_argument("--input", type=Path, help="Generated data directory or ZIP archive")
    parser.add_argument("--output", type=Path, help="Directory for completed runs")
    parser.add_argument("--include-late", action="store_true", help="Load late deliveries")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    with args.config.open(encoding="utf-8") as source:
        config = json.load(source)
    config_dir = args.config.resolve().parent
    default_input = config.pop("input_path", "../1st_task/out")
    default_output = config.pop("output_path", "artifacts")
    input_path = args.input or config_dir / default_input
    output_path = args.output or config_dir / default_output
    print(run_pipeline(input_path.resolve(), output_path.resolve(), config, args.include_late))


if __name__ == "__main__":
    main()

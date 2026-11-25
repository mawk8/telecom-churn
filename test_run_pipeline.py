import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_pipeline import ARTIFACTS, completed, file_digest, fingerprint, run_pipeline


class PipelinePublicationTests(unittest.TestCase):
    def test_fingerprint_tracks_input_contents_and_configuration(self):
        root = Path(tempfile.mkdtemp(prefix="churn-fingerprint-"))
        source = root / "source.csv"
        source.write_text("first", encoding="utf-8")
        initial = fingerprint(root, [source], {"seed": 42}, False)
        self.assertEqual(initial, fingerprint(root, [source], {"seed": 42}, False))
        self.assertNotEqual(initial, fingerprint(root, [source], {"seed": 43}, False))
        self.assertNotEqual(initial, fingerprint(root, [source], {"seed": 42}, True))
        source.write_text("other", encoding="utf-8")
        self.assertNotEqual(initial, fingerprint(root, [source], {"seed": 42}, False))

    def test_marker_and_artifacts_are_required(self):
        root = Path(tempfile.mkdtemp(prefix="churn-publication-"))
        for artifact in ARTIFACTS:
            (root / artifact).write_text("artifact", encoding="utf-8")
        marker = root / "_SUCCESS.json"
        marker.write_text("{}", encoding="utf-8")
        self.assertFalse(completed(root))
        manifest = {
            "run_id": root.name,
            "artifact_sha256": {name: file_digest(root / name) for name in ARTIFACTS},
        }
        marker.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertTrue(completed(root))
        (root / ARTIFACTS[0]).write_text("modified", encoding="utf-8")
        self.assertFalse(completed(root))

    def test_incomplete_artifacts_remain_unpublished_and_can_be_retried(self):
        root = Path(tempfile.mkdtemp(prefix="churn-retry-"))
        source = root / "source.csv"
        source.write_text("data", encoding="utf-8")
        output = root / "output"

        def create_artifacts(staging, *_):
            artifacts = ARTIFACTS if first_attempt.exists() else ARTIFACTS[:-1]
            for artifact in artifacts:
                (staging / artifact).write_text("artifact", encoding="utf-8")
            first_attempt.touch()

        first_attempt = root / "attempted"
        with (
            patch("features.input_files", return_value=(source,)),
            patch("run_pipeline.runtime", return_value={}),
            patch("run_pipeline.build_artifacts", side_effect=create_artifacts) as build,
        ):
            with self.assertRaises(RuntimeError):
                run_pipeline(root, output, {})
            self.assertEqual(len(list(output.glob(".pending-*"))), 1)
            destination = run_pipeline(root, output, {})
            self.assertTrue(completed(destination))
            self.assertEqual(run_pipeline(root, output, {}), destination)
            damaged = destination / ARTIFACTS[0]
            damaged.write_text("damaged", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run_pipeline(root, output, {})
            self.assertEqual(damaged.read_text(encoding="utf-8"), "damaged")
            self.assertEqual(build.call_count, 2)


if __name__ == "__main__":
    unittest.main()

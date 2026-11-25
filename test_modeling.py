import io
import unittest

import joblib
import numpy as np
import pandas as pd


class ScoreEstimator:
    def predict_proba(self, features):
        scores = features["score"].to_numpy()
        return np.column_stack((1 - scores, scores))


class ModelingTests(unittest.TestCase):
    def test_selects_candidate_and_preserves_predictions_after_serialization(self):
        from modeling import evaluate, train_model

        frame = pd.DataFrame({
            "subscriber_key": [f"id-{index:03}" for index in range(100)],
            "label": [index % 5 == 0 for index in range(100)],
        })
        frame["score"] = 0.0
        frame["tel_signal"] = frame["label"].astype(float)
        frame.loc[1, "tel_signal"] = np.nan
        bundle, validation = train_model(frame, {"xgb_estimators": 5, "threads": 1})
        self.assertEqual(len(validation["candidates"]), 4)
        self.assertEqual(validation["fit_count"], 80)
        self.assertEqual(validation["count"], 20)
        self.assertEqual(validation["positive_count"], 4)
        self.assertIn("tel_signal", bundle["features"])
        self.assertEqual(bundle["name"], validation["selected_model"])
        self.assertNotIn("subscriber_key", bundle["features"])
        self.assertNotIn("label", bundle["features"])

        handle = io.BytesIO()
        joblib.dump(bundle, handle)
        handle.seek(0)
        restored = joblib.load(handle)
        self.assertEqual(evaluate(bundle, frame), evaluate(restored, frame))
        _, reordered_validation = train_model(
            frame.iloc[::-1], {"xgb_estimators": 5, "threads": 1}
        )
        self.assertEqual(validation, reordered_validation)

    def test_calculates_metrics_and_breaks_score_ties_by_subscriber(self):
        from modeling import evaluate

        frame = pd.DataFrame({
            "subscriber_key": [f"id-{index:02}" for index in range(21)],
            "label": [1, 0, 1] + [0] * 18,
            "score": [0.8, 0.8, 0.8] + [0.2] * 18,
        })
        bundle = {"estimator": ScoreEstimator(), "features": ["score"], "threshold": 0.8}
        result = evaluate(bundle, frame.iloc[::-1])
        self.assertEqual(result["count"], 21)
        self.assertEqual(result["positive_count"], 2)
        self.assertAlmostEqual(result["prevalence"], 2 / 21)
        self.assertAlmostEqual(result["average_precision"], 2 / 3)
        self.assertAlmostEqual(result["roc_auc"], 37 / 38)
        self.assertAlmostEqual(result["f1"], 0.8)
        self.assertEqual(result["precision_at_5_percent"], 0.5)
        self.assertEqual(result["top_count"], 2)
        self.assertEqual(result["top_positive_count"], 1)

    def test_rejects_missing_class(self):
        from modeling import evaluate, train_model

        frame = pd.DataFrame({"subscriber_key": ["a"], "label": [0], "score": [0.5]})
        with self.assertRaisesRegex(ValueError, "both"):
            train_model(frame, {})
        with self.assertRaisesRegex(ValueError, "both"):
            evaluate({}, frame)

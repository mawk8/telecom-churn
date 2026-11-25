"""Select on the early cohort and score the selected model on a later cohort."""

from math import ceil

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def _labels(frame, cohort):
    if frame["label"].isna().any() or set(frame["label"].unique()) != {0, 1}:
        raise ValueError(f"The {cohort} cohort must contain both binary classes (0 and 1).")
    return frame["label"].astype(int)


def _counts(labels):
    return {
        "count": len(labels),
        "positive_count": int(labels.sum()),
        "prevalence": float(labels.mean()),
    }


def _best_threshold(labels, scores):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # The final precision/recall pair has no corresponding threshold.
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) > 0,
    )
    # Prefer a higher threshold when several thresholds give the same F1.
    best = np.flatnonzero(f1 == f1.max())[-1]
    return float(thresholds[best]), float(f1[best])


def train_model(train_frame, cfg):
    """Select by validation AP. Reuse the fitted winner so its threshold stays fixed."""
    early = train_frame.sort_values("subscriber_key").reset_index(drop=True)
    labels = _labels(early, "early")
    if early["subscriber_key"].isna().any() or early["subscriber_key"].duplicated().any():
        raise ValueError("The early cohort must contain one row per nonmissing subscriber key.")

    seed = cfg.get("seed", 42)
    fit_rows, val_rows = train_test_split(
        early,
        test_size=cfg.get("validation_fraction", 0.2),
        random_state=seed,
        stratify=labels,
    )
    fit_labels = _labels(fit_rows, "training")
    val_labels = _labels(val_rows, "validation")

    all_features = sorted(set(early.columns) - {"subscriber_key", "label"})
    base_features = [name for name in all_features if not name.startswith("tel_")]
    if not base_features:
        raise ValueError("The early cohort must contain at least one base feature.")

    feature_sets = [("without_telemetry", base_features)]
    if all_features != base_features:
        feature_sets.append(("with_telemetry", all_features))

    candidates = []
    best_bundle = None
    best_ap = -1.0
    for suffix, features in feature_sets:
        estimators = [
            ("logistic_regression", make_pipeline(
                SimpleImputer(strategy="median", keep_empty_features=True),
                StandardScaler(),
                LogisticRegression(
                    max_iter=cfg.get("logistic_max_iter", 500),
                    random_state=seed,
                ),
            )),
            ("xgboost", XGBClassifier(
                n_estimators=cfg.get("xgb_estimators", 100),
                max_depth=cfg.get("xgb_max_depth", 4),
                n_jobs=cfg.get("threads", 4),
                random_state=seed,
                eval_metric="logloss",
            )),
        ]
        for model_name, estimator in estimators:
            name = f"{model_name}_{suffix}"
            estimator.fit(fit_rows[features], fit_labels)
            scores = estimator.predict_proba(val_rows[features])[:, 1]
            ap = float(average_precision_score(val_labels, scores))
            threshold, f1 = _best_threshold(val_labels, scores)
            candidates.append({
                "name": name,
                "average_precision": ap,
                "f1": f1,
                "threshold": threshold,
                "feature_count": len(features),
            })
            # The first match wins ties, preferring the base feature set.
            if ap > best_ap:
                best_ap = ap
                best_bundle = {
                    "name": name,
                    "estimator": estimator,
                    "features": features,
                    "threshold": threshold,
                }

    validation = {
        **_counts(val_labels),
        "fit_count": len(fit_labels),
        "fit_positive_count": int(fit_labels.sum()),
        "candidates": candidates,
        "selected_model": best_bundle["name"],
        "threshold": best_bundle["threshold"],
    }
    return best_bundle, validation


def evaluate(bundle, test_frame):
    """Score a later cohort using the saved estimator, features, and threshold."""
    ordered = test_frame.sort_values("subscriber_key")
    labels = _labels(ordered, "test")
    scores = bundle["estimator"].predict_proba(ordered[bundle["features"]])[:, 1]
    top_count = ceil(len(labels) * 0.05)
    # A stable sort preserves subscriber-key order within equal scores.
    top_rows = np.argsort(-scores, kind="stable")[:top_count]
    top_positive_count = int(labels.iloc[top_rows].sum())
    return {
        **_counts(labels),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "f1": float(f1_score(labels, scores >= bundle["threshold"], zero_division=0)),
        "precision_at_5_percent": top_positive_count / top_count,
        "top_count": top_count,
        "top_positive_count": top_positive_count,
    }

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import checksum
from .features import build_features


def estimator(name: str, seed: int):
    if name == "logistic_regression":
        return Pipeline(
            [("scale", StandardScaler()), ("model", LogisticRegression(max_iter=3000, C=0.5))]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=280, min_samples_leaf=7, max_features=0.75, n_jobs=-1, random_state=seed
        )
    if name == "histogram_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=25,
            l2_regularization=0.4,
            random_state=seed,
        )
    raise ValueError(name)


def metrics(y: pd.Series, p: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
    }


def train(config: dict[str, Any]) -> dict[str, Any]:
    frame = build_features(config)
    features = config["features"]["columns"]
    target = config["features"]["target"]
    test_start = pd.Timestamp(config["training"]["test_start"])
    train_frame = frame[frame.first_contribution < test_start]
    test = frame[frame.first_contribution >= test_start]
    split = train_frame.first_contribution.quantile(0.8)
    fit = train_frame[train_frame.first_contribution < split]
    validation = train_frame[train_frame.first_contribution >= split]
    results = {}
    seed = config["project"]["seed"]
    for name in config["training"]["candidates"]:
        model = estimator(name, seed)
        model.fit(fit[features], fit[target])
        p = model.predict_proba(validation[features])[:, 1]
        results[name] = metrics(validation[target], p, config["training"]["review_threshold"])
    selected = max(results, key=lambda name: results[name]["average_precision"])
    model = estimator(selected, seed)
    model.fit(train_frame[features], train_frame[target])
    probability = model.predict_proba(test[features])[:, 1]
    anomaly = IsolationForest(n_estimators=200, contamination=0.05, random_state=seed)
    anomaly.fit(train_frame[features])
    output = test[["donor_id", "is_sybil", "first_contribution"]].copy()
    output["risk_probability"] = probability
    output["anomaly"] = anomaly.predict(test[features]) == -1
    review, block = config["training"]["review_threshold"], config["training"]["block_threshold"]
    output["action"] = np.where(
        probability >= block,
        "block_pending_appeal",
        np.where(probability >= review, "human_review", "allow"),
    )
    metadata = {
        "version": config["project"]["version"],
        "selected_model": selected,
        "features": features,
        "candidate_validation": results,
        "test_metrics": metrics(test[target], probability, review),
        "thresholds": {"review": review, "block": block},
        "action_rates": output.action.value_counts(normalize=True).to_dict(),
        "train_rows": len(train_frame),
        "test_rows": len(test),
        "data_sha256": checksum(config["paths"]["features"]),
    }
    artifacts, reports = Path(config["paths"]["artifacts"]), Path(config["paths"]["reports"])
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifacts / "sybil_model.joblib")
    joblib.dump(anomaly, artifacts / "anomaly_model.joblib")
    (artifacts / "metadata.json").write_text(json.dumps(metadata, indent=2))
    output.to_csv(reports / "test_predictions.csv", index=False)
    full_probability = model.predict_proba(frame[features])[:, 1]
    frame[["donor_id"]].assign(risk_probability=full_probability).to_csv(
        artifacts / "donor_scores.csv", index=False
    )
    return metadata


def load_artifacts(path: str | Path):
    root = Path(path)
    return (
        joblib.load(root / "sybil_model.joblib"),
        joblib.load(root / "anomaly_model.joblib"),
        json.loads((root / "metadata.json").read_text()),
    )

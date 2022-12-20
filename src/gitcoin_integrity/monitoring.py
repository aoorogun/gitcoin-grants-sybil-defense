from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def psi(a: pd.Series, b: pd.Series) -> float:
    edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    x = np.clip(np.histogram(a, edges)[0] / len(a), 1e-6, None)
    y = np.clip(np.histogram(b, edges)[0] / len(b), 1e-6, None)
    return float(np.sum((y - x) * np.log(y / x)))


def monitor(config: dict[str, Any]) -> dict[str, Any]:
    frame = pd.read_csv(config["paths"]["features"], parse_dates=["first_contribution"])
    split = frame.first_contribution.quantile(0.7)
    a, b = frame[frame.first_contribution < split], frame[frame.first_contribution >= split]
    scores = {c: psi(a[c], b[c]) for c in config["features"]["columns"]}
    maximum = max(scores.values())
    m = config["monitoring"]
    status = (
        "critical"
        if maximum >= m["psi_critical"]
        else "warning"
        if maximum >= m["psi_warning"]
        else "healthy"
    )
    result = {"status": status, "maximum_psi": maximum, "feature_psi": scores}
    (Path(config["paths"]["reports"]) / "drift_report.json").write_text(
        json.dumps(result, indent=2)
    )
    return result

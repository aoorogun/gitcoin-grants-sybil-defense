from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import load_tables


def qf_allocate(
    contributions: pd.DataFrame, pool: float, donor_weight: pd.Series | None = None
) -> pd.DataFrame:
    frame = contributions.copy()
    frame["weight"] = 1.0 if donor_weight is None else frame.donor_id.map(donor_weight).fillna(1.0)
    frame["weighted_amount"] = frame.amount_usd * frame.weight
    raw = frame.groupby("grant_id").weighted_amount.apply(
        lambda values: float(np.square(np.sqrt(values).sum()))
    )
    direct = frame.groupby("grant_id").amount_usd.sum()
    result = pd.DataFrame({"direct_usd": direct, "qf_score": raw})
    result["matching_usd"] = result.qf_score / result.qf_score.sum() * pool
    return result.reset_index().sort_values("matching_usd", ascending=False)


def allocate(config: dict[str, Any]) -> pd.DataFrame:
    _, contributions = load_tables(config)
    scores = pd.read_csv(Path(config["paths"]["artifacts"]) / "donor_scores.csv")
    block = config["training"]["block_threshold"]
    weights = scores.set_index("donor_id").risk_probability.map(
        lambda p: 0.0 if p >= block else 1.0
    )
    result = qf_allocate(contributions, config["allocation"]["matching_pool_usd"], weights)
    result.to_csv(Path(config["paths"]["reports"]) / "matching_allocations.csv", index=False)
    return result

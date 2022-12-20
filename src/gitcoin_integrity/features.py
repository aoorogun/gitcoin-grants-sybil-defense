from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .data import load_tables


def build_features(config: dict[str, Any]) -> pd.DataFrame:
    donors, contributions = load_tables(config)
    contributions = contributions.sort_values(["donor_id", "timestamp"])
    contributions["hour"] = contributions.timestamp.dt.hour
    contributions["is_night"] = contributions.hour.between(0, 5)
    contributions["gap_minutes"] = (
        contributions.groupby("donor_id").timestamp.diff().dt.total_seconds().div(60)
    )
    contributions["is_burst"] = contributions.gap_minutes.lt(2)
    grouped = contributions.groupby("donor_id")
    aggregate = grouped.agg(
        contribution_count=("contribution_id", "count"),
        unique_grants=("grant_id", "nunique"),
        total_usd=("amount_usd", "sum"),
        mean_usd=("amount_usd", "mean"),
        std_usd=("amount_usd", "std"),
        burst_share=("is_burst", "mean"),
        night_share=("is_night", "mean"),
        first_contribution=("timestamp", "min"),
    )
    top = contributions.groupby(["donor_id", "grant_id"]).amount_usd.sum().groupby("donor_id").max()
    aggregate["top_grant_share"] = top / aggregate.total_usd
    aggregate["amount_cv"] = aggregate.std_usd.fillna(0) / aggregate.mean_usd.clip(lower=0.01)
    group_sizes = donors.groupby("fingerprint_group").size()
    donors["fingerprint_group_size"] = donors.fingerprint_group.map(group_sizes)
    result = donors.merge(aggregate.reset_index(), on="donor_id", how="left").fillna(
        {
            "contribution_count": 0,
            "unique_grants": 0,
            "total_usd": 0,
            "mean_usd": 0,
            "std_usd": 0,
            "burst_share": 0,
            "night_share": 0,
            "top_grant_share": 0,
            "amount_cv": 0,
        }
    )
    result["first_contribution"] = pd.to_datetime(result.first_contribution).fillna(
        pd.Timestamp(config["data"]["round_start"])
    )
    path = Path(config["paths"]["features"])
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False)
    return result

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .allocation import qf_allocate
from .data import load_tables


def run_scenario(config: dict[str, Any], path: str | Path) -> dict[str, Any]:
    scenario = json.loads(Path(path).read_text())
    _, contributions = load_tables(config)
    scores = (
        pd.read_csv(Path(config["paths"]["artifacts"]) / "donor_scores.csv")
        .set_index("donor_id")
        .risk_probability
    )
    base_block = config["training"]["block_threshold"]
    base_weights = scores.map(lambda p: 0.0 if p >= base_block else 1.0)
    review, block = scenario["review_threshold"], scenario["block_threshold"]
    scenario_weights = scores.map(
        lambda p: 0.0 if p >= block else scenario["downweight_reviewed"] if p >= review else 1.0
    )
    pool = config["allocation"]["matching_pool_usd"]
    baseline = qf_allocate(contributions, pool, base_weights)
    changed = qf_allocate(contributions, pool, scenario_weights)
    comparison = baseline[["grant_id", "matching_usd"]].merge(
        changed[["grant_id", "matching_usd"]], on="grant_id", suffixes=("_baseline", "_scenario")
    )
    comparison["delta_usd"] = comparison.matching_usd_scenario - comparison.matching_usd_baseline
    comparison.to_csv(Path(config["paths"]["reports"]) / "scenario_allocations.csv", index=False)
    result = {
        "scenario_id": scenario["scenario_id"],
        "reviewed_donors": int(((scores >= review) & (scores < block)).sum()),
        "blocked_donors": int((scores >= block).sum()),
        "grants_changed_over_1000_usd": int((comparison.delta_usd.abs() >= 1000).sum()),
        "largest_absolute_reallocation_usd": float(comparison.delta_usd.abs().max()),
        "pool_reconciles": bool(abs(changed.matching_usd.sum() - pool) < 0.01),
    }
    (Path(config["paths"]["reports"]) / "scenario_result.json").write_text(
        json.dumps(result, indent=2)
    )
    return result

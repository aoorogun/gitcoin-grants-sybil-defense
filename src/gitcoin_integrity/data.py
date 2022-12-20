from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def generate(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(config["project"]["seed"])
    dcfg = config["data"]
    n, grants = dcfg["donors"], dcfg["grants"]
    latent = rng.random(n) < 0.28
    donors = pd.DataFrame({"donor_id": [f"0x{i:040x}" for i in range(n)]})
    donors["wallet_age_days"] = np.where(latent, rng.integers(1, 90, n), rng.integers(30, 1800, n))
    donors["prior_tx_count"] = np.where(
        latent, rng.poisson(4, n), rng.negative_binomial(4, 0.12, n)
    )
    donors["trust_score"] = np.clip(
        np.where(latent, rng.beta(1.5, 7, n), rng.beta(5, 2.5, n)), 0, 1
    )
    group = np.arange(n)
    sybil_indices = np.flatnonzero(latent)
    group[sybil_indices] = rng.integers(0, max(1, len(sybil_indices) // 8), len(sybil_indices))
    donors["fingerprint_group"] = [f"fp_{x:06d}" for x in group]
    # Simulate imperfect human/heuristic adjudication rather than treating a
    # generated behavioural rule as infallible ground truth.
    donors["is_sybil"] = np.logical_xor(latent, rng.random(n) < 0.10).astype(int)
    weights = np.where(latent, rng.gamma(7, 1, n), rng.gamma(4, 1, n))
    weights /= weights.sum()
    chosen = rng.choice(n, dcfg["contributions"], p=weights)
    grant_popularity = rng.zipf(1.7, dcfg["contributions"]) % grants
    start = pd.Timestamp(dcfg["round_start"])
    offsets = rng.integers(0, dcfg["round_days"] * 86400, dcfg["contributions"])
    contributions = pd.DataFrame(
        {
            "contribution_id": [f"c_{i:08d}" for i in range(dcfg["contributions"])],
            "donor_id": donors.loc[chosen, "donor_id"].to_numpy(),
            "grant_id": [f"grant_{i:04d}" for i in grant_popularity],
            "timestamp": start + pd.to_timedelta(offsets, unit="s"),
            "amount_usd": np.round(
                np.maximum(1, rng.lognormal(2.4, 0.9, dcfg["contributions"])), 2
            ),
        }
    )
    sybil_contribution = latent[chosen]
    contributions.loc[sybil_contribution, "amount_usd"] = np.round(
        rng.uniform(1, 8, sybil_contribution.sum()), 2
    )
    for frame, key in [(donors, "donors"), (contributions, "contributions")]:
        path = Path(config["paths"][key])
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    return donors, contributions


def load_tables(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    donors = pd.read_csv(config["paths"]["donors"])
    contributions = pd.read_csv(config["paths"]["contributions"], parse_dates=["timestamp"])
    if donors.donor_id.duplicated().any() or contributions.contribution_id.duplicated().any():
        raise ValueError("Duplicate IDs")
    if (contributions.amount_usd <= 0).any():
        raise ValueError("Contribution amounts must be positive")
    return donors, contributions


def checksum(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

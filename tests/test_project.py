from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from gitcoin_integrity.allocation import qf_allocate
from gitcoin_integrity.config import load_config
from gitcoin_integrity.data import generate
from gitcoin_integrity.features import build_features
from gitcoin_integrity.modeling import train
from gitcoin_integrity.monitoring import psi


class ProjectTests(unittest.TestCase):
    def test_qf_reconciles_pool(self):
        data = pd.DataFrame(
            {
                "donor_id": ["a", "b", "a"],
                "grant_id": ["x", "x", "y"],
                "amount_usd": [4.0, 9.0, 16.0],
            }
        )
        result = qf_allocate(data, 1000)
        self.assertAlmostEqual(result.matching_usd.sum(), 1000)

    def test_generation_features_and_training(self):
        config = load_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config["data"].update({"donors": 1200, "grants": 50, "contributions": 7000})
            for key, name in [
                ("donors", "donors.csv"),
                ("contributions", "contributions.csv"),
                ("features", "features.csv"),
                ("artifacts", "artifacts"),
                ("reports", "reports"),
            ]:
                config["paths"][key] = str(root / name)
            config["training"]["candidates"] = ["logistic_regression"]
            generate(config)
            features = build_features(config)
            self.assertEqual(len(features), 1200)
            metadata = train(config)
            self.assertIn("average_precision", metadata["test_metrics"])

    def test_psi_detects_shift(self):
        rng = np.random.default_rng(4)
        self.assertGreater(
            psi(pd.Series(rng.normal(0, 1, 2000)), pd.Series(rng.normal(2, 1, 2000))), 0.25
        )


if __name__ == "__main__":
    unittest.main()

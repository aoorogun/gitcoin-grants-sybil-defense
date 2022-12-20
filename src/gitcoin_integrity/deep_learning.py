from __future__ import annotations

from typing import Any


def build_donor_network(feature_dim: int, grants: int) -> Any:
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("Install .[deep-learning]") from exc

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.grant_embedding = nn.EmbeddingBag(grants, 16, mode="mean")
            self.network = nn.Sequential(
                nn.Linear(feature_dim + 16, 96), nn.ReLU(), nn.Dropout(0.25), nn.Linear(96, 1)
            )

        def forward(self, features, grant_ids, offsets):
            return self.network(
                __import__("torch").cat([features, self.grant_embedding(grant_ids, offsets)], dim=1)
            )

    return Model()

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from .config import load_config
from .modeling import load_artifacts


class DonorFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contribution_count: float
    unique_grants: float
    total_usd: float
    mean_usd: float
    amount_cv: float
    top_grant_share: float
    burst_share: float
    night_share: float
    wallet_age_days: float
    prior_tx_count: float
    trust_score: float
    fingerprint_group_size: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = load_config(os.getenv("GI_CONFIG"))
    root = os.getenv("GI_MODEL_DIR", config["paths"]["artifacts"])
    try:
        app.state.model, app.state.anomaly, app.state.metadata = load_artifacts(root)
        app.state.ready = True
    except FileNotFoundError:
        app.state.ready = False
    yield


app = FastAPI(title="Gitcoin Grants Integrity API", version="1.0.0", lifespan=lifespan)


@app.get("/health/ready")
def ready(request: Request) -> dict[str, Any]:
    if not request.app.state.ready:
        raise HTTPException(503, "Model unavailable")
    return {"status": "ready", "version": request.app.state.metadata["version"]}


@app.post("/v1/risk")
def risk(payload: DonorFeatures, request: Request) -> dict[str, Any]:
    if not request.app.state.ready:
        raise HTTPException(503, "Model unavailable")
    frame = pd.DataFrame([payload.model_dump()])
    metadata = request.app.state.metadata
    probability = float(request.app.state.model.predict_proba(frame[metadata["features"]])[0, 1])
    thresholds = metadata["thresholds"]
    action = (
        "block_pending_appeal"
        if probability >= thresholds["block"]
        else "human_review"
        if probability >= thresholds["review"]
        else "allow"
    )
    anomaly = bool(request.app.state.anomaly.predict(frame[metadata["features"]])[0] == -1)
    return {
        "risk_probability": probability,
        "recommended_action": action,
        "anomaly": anomaly,
        "model_version": metadata["version"],
    }

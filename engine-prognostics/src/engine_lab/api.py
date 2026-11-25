"""HTTP inference service for a single registered C-MAPSS fleet model."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .data import COLUMNS, read_trajectory
from .monitoring import assess_drift


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[dict[str, float]] = Field(min_length=1, description="Observed engine-cycle rows using the 26 C-MAPSS column names")


def create_app(model_path: str | Path) -> FastAPI:
    import joblib

    model = joblib.load(model_path)
    app = FastAPI(
        title="Engine RUL Inference API",
        version="1.0.0",
        description="Research demonstration for remaining useful life estimates on simulated C-MAPSS trajectories.",
    )
    app.state.predictor = model

    @app.get("/health")
    def health():
        return {
            "status": "ready",
            "subset": model.subset,
            "model": model.model_name,
            "data_kind": "simulated",
            "monitoring": "PSI on engineered input features; heuristic alert threshold",
        }

    @app.get("/schema")
    def schema():
        return {"columns_in_order": COLUMNS, "record_count": "one or more observed cycles from one or more engines"}

    @app.post("/predict")
    def predict(request: PredictionRequest):
        records = request.records
        if any(set(row) != set(COLUMNS) for row in records):
            raise HTTPException(
                status_code=422,
                detail={"message": "Each record must contain exactly the 26 C-MAPSS columns; see /schema"},
            )
        frame = pd.DataFrame(records, columns=COLUMNS)
        try:
            text = io.StringIO()
            np.savetxt(text, frame.to_numpy(dtype=float))
            text.seek(0)
            frame = read_trajectory(text)
            predicted, lower, upper = model.terminal(frame)
            drift = assess_drift(
                model.reference_profile,
                model.features.transform(frame),
                model.features.feature_names,
            )
        except (ValueError, TypeError, KeyError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        terminal = frame.groupby("engine", sort=True).tail(1)
        estimates = [
            {
                "engine": int(row.engine),
                "last_observed_cycle": int(row.cycle),
                "predicted_rul_cycles": float(predicted[index]),
                "interval_lower_cycles": float(lower[index]),
                "interval_upper_cycles": float(upper[index]),
            }
            for index, row in enumerate(terminal.itertuples())
        ]
        return {"subset": model.subset, "estimates": estimates, "input_drift": drift}

    return app

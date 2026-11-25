"""Experiment tracking, validation gate and model-registry integration."""

from __future__ import annotations

import math
from pathlib import Path

def promotion_gate(metrics):
    """Promote only a non-baseline candidate that beats the age baseline on validation."""
    validation = metrics["validation"]
    baseline = validation["Age baseline"]["mae"]
    selected = validation[metrics["selected_model"]]["mae"]
    passed = (
        metrics["selected_model"] != "Age baseline"
        and math.isfinite(baseline)
        and math.isfinite(selected)
        and selected < baseline
    )
    return {
        "passed": passed,
        "decision_metric": "engine-separated validation endpoint MAE",
        "selected_mae": selected,
        "age_baseline_mae": baseline,
        "reason": "candidate beats baseline" if passed else "no non-baseline improvement",
    }


def log_training_run(*, subset, model, model_path, metrics, seed, rul_cap, archive_sha256):
    """Record lineage and register only models that pass the validation-only gate."""
    import mlflow
    import mlflow.pyfunc
    import pandas as pd
    from mlflow.models import ModelSignature
    from mlflow.types import ColSpec, DataType, Schema

    from .data import COLUMNS

    gate = metrics["promotion_gate"]
    signature = ModelSignature(
        inputs=Schema([ColSpec(DataType.double, name) for name in COLUMNS]),
        outputs=Schema(
            [ColSpec(DataType.double, name) for name in ("rul", "lower", "upper")]
        ),
    )
    input_example = pd.DataFrame([[1.0, 1.0, *([0.0] * 24)]], columns=COLUMNS)
    with mlflow.start_run(run_name=f"{subset}-{metrics['selected_model']}"):
        mlflow.log_params(
            {
                "subset": subset,
                "selected_model": metrics["selected_model"],
                "seed": seed,
                "rul_cap": rul_cap,
                "regimes": metrics["regimes"],
                "feature_count": metrics["feature_count"],
                "data_kind": "simulated",
                "archive_sha256": archive_sha256,
            }
        )
        mlflow.log_metrics(
            {
                "validation_selected_mae": gate["selected_mae"],
                "validation_age_baseline_mae": gate["age_baseline_mae"],
                "official_test_rmse_diagnostic": metrics["test"]["rmse"],
                "official_test_mae_diagnostic": metrics["test"]["mae"],
                "test_interval_coverage_diagnostic": metrics["interval"]["empirical_test_coverage"],
            }
        )
        mlflow.set_tags(
            {
                "promotion_gate": "passed" if gate["passed"] else "blocked",
                "test_metrics_used_for_promotion": "false",
                "data_status": "simulated_benchmark",
            }
        )
        mlflow.log_artifact(str(model_path), artifact_path="joblib")
        if gate["passed"]:
            mlflow.pyfunc.log_model(
                name="registered_predictor",
                python_model=str(Path(__file__).with_name("mlflow_model.py")),
                artifacts={"predictor": str(Path(model_path).resolve())},
                registered_model_name=f"engine-rul-{subset.lower()}",
                signature=signature,
                input_example=input_example,
                code_paths=[str(Path(__file__).resolve().parents[1])],
                pip_requirements=[
                    "numpy==2.5.5",
                    "pandas==3.1.1",
                    "scikit-learn==1.9.1",
                    "joblib==1.6.0",
                    "threadpoolctl>=3.5,<4",
                ],
            )

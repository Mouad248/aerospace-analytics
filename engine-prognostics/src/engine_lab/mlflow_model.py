"""MLflow models-from-code entry point for a registered RUL predictor."""

from __future__ import annotations

from typing import Any

import mlflow
import pandas as pd


class RegisteredRULModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        import joblib

        self.predictor = joblib.load(context.artifacts["predictor"])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        predicted, lower, upper = self.predictor.terminal(model_input)
        return pd.DataFrame({"rul": predicted, "lower": lower, "upper": upper})


mlflow.models.set_model(RegisteredRULModel())

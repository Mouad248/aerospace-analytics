from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .data import last_indices, pseudo_endpoints, split_engines, training_rul
from .features import FeatureBuilder
from .monitoring import build_reference_profile


def metrics(truth, prediction):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if (
        truth.ndim != 1
        or truth.shape != prediction.shape
        or not len(truth)
        or not np.isfinite(truth).all()
        or not np.isfinite(prediction).all()
    ):
        raise ValueError("Metric arrays must be aligned, nonempty and finite")
    error = prediction - truth
    return {
        "engines": len(truth),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_error": float(error.mean()),
        "asymmetric_score": float(
            np.sum(np.expm1(np.where(error < 0, -error / 13, error / 10)))
        ),
    }


def conformal_radius(residuals, alpha=0.2):
    residuals = np.asarray(residuals, dtype=float)
    if (
        residuals.ndim != 1
        or not len(residuals)
        or not np.isfinite(residuals).all()
        or np.any(residuals < 0)
        or not 0 < alpha < 1
    ):
        raise ValueError(
            "Finite nonnegative calibration scores and alpha in (0,1) required"
        )
    rank = math.ceil((len(residuals) + 1) * (1 - alpha))
    if rank > len(residuals):
        raise ValueError("Too few calibration engines for the requested coverage")
    return float(np.sort(residuals)[rank - 1])


@dataclass
class Predictor:
    features: FeatureBuilder
    estimator: object
    model_name: str
    median_lifetime: float
    cap: float
    radius: float = 0
    subset: str = ""

    def predict_all(self, frame):
        if self.model_name == "Age baseline":
            values = self.median_lifetime - frame["cycle"].to_numpy()
        else:
            with threadpool_limits(limits=2):
                values = self.estimator.predict(self.features.transform(frame))
        return np.clip(values, 0, self.cap)

    def terminal(self, frame):
        indices = last_indices(frame)
        prediction = self.predict_all(frame)[indices]
        return (
            prediction,
            np.maximum(0, prediction - self.radius),
            prediction + self.radius,
        )


def review_metrics(truth, lower, horizon=30):
    positive, flag = np.asarray(truth) <= horizon, np.asarray(lower) <= horizon
    tp, fp = int(np.sum(positive & flag)), int(np.sum(~positive & flag))
    fn, tn = int(np.sum(positive & ~flag)), int(np.sum(~positive & ~flag))
    return {
        "horizon_cycles": horizon,
        "flagged": tp + fp,
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def train_subset(train, test, test_truth, subset, seed=2026, cap=125, alpha=0.2):
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("Positive finite RUL training cap required")
    split = split_engines(train, seed)
    groups = {
        name: train[train["engine"].isin(ids)].reset_index(drop=True)
        for name, ids in split.items()
    }
    fit = groups["fit"]
    with threadpool_limits(limits=2):
        features = FeatureBuilder(
            6 if subset in {"FD002", "FD004"} else 1, seed=seed
        ).fit(fit)
        x = features.transform(fit)
        target = np.minimum(training_rul(fit), cap)
        # Equal total weight per engine prevents long trajectories dominating fit.
        weights = 1 / fit.groupby("engine")["engine"].transform("size").to_numpy()
        weights *= len(weights) / weights.sum()
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=50))
        ridge.fit(x, target, ridge__sample_weight=weights)
        boost = HistGradientBoostingRegressor(
            max_iter=120,
            max_leaf_nodes=15,
            learning_rate=0.06,
            l2_regularization=2,
            min_samples_leaf=30,
            early_stopping=False,
            random_state=seed,
        )
        boost.fit(x, target, sample_weight=weights)
    median_lifetime = float(fit.groupby("engine")["cycle"].max().median())
    candidates = {
        "Age baseline": Predictor(
            features, None, "Age baseline", median_lifetime, cap, subset=subset
        ),
        "Ridge": Predictor(
            features, ridge, "Ridge", median_lifetime, cap, subset=subset
        ),
        "Gradient boosting": Predictor(
            features, boost, "Gradient boosting", median_lifetime, cap, subset=subset
        ),
    }
    validation = groups["validation"]
    endpoints = pseudo_endpoints(validation, seed)
    truth = training_rul(validation)[endpoints]
    selection = {
        name: metrics(truth, model.predict_all(validation)[endpoints])
        for name, model in candidates.items()
    }
    selected = min(selection, key=lambda name: (selection[name]["mae"], name))
    model = candidates[selected]
    model.reference_profile = build_reference_profile(
        x, features.feature_names
    )
    calibration = groups["calibration"]
    cal_indices = pseudo_endpoints(calibration, seed)
    cal_truth = training_rul(calibration)[cal_indices]
    cal_prediction = model.predict_all(calibration)[cal_indices]
    model.radius = conformal_radius(np.abs(cal_truth - cal_prediction), alpha)
    prediction, lower, upper = model.terminal(test)
    results = {
        "subset": subset,
        "data_kind": "NASA simulated degradation",
        "seed": seed,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_engines": train["engine"].nunique(),
        "test_engines": test["engine"].nunique(),
        "split": split,
        "regimes": features.regimes,
        "feature_count": len(features.feature_names),
        "window": features.window,
        "training_rul_cap": cap,
        "validation_endpoint_fraction": [0.35, 0.85],
        "selection_metric": "validation engine-endpoint MAE on uncapped RUL",
        "selected_model": selected,
        "validation": selection,
        "test_comparison": {
            name: metrics(test_truth, m.terminal(test)[0])
            for name, m in candidates.items()
        },
        "test": metrics(test_truth, prediction),
        "test_clipped_target": metrics(np.minimum(test_truth, cap), prediction),
        "interval": {
            "nominal_coverage": 1 - alpha,
            "radius": model.radius,
            "calibration_engines": len(cal_truth),
            "empirical_test_coverage": float(
                np.mean((test_truth >= lower) & (test_truth <= upper))
            ),
            "mean_width": float(np.mean(upper - lower)),
        },
        "review_screen": review_metrics(test_truth, lower),
        "feature_names": features.feature_names,
    }
    predictions = []
    for i, row in enumerate(test.iloc[last_indices(test)].itertuples()):
        predictions.append(
            {
                "engine": int(row.engine),
                "cycle": int(row.cycle),
                "rul_actual": float(test_truth[i]),
                "rul_predicted": float(prediction[i]),
                "lower": float(lower[i]),
                "upper": float(upper[i]),
            }
        )
    trajectories = []
    all_predictions = model.predict_all(test)
    for engine in sorted(test["engine"].unique())[:3]:
        frame = test[test["engine"] == engine]
        actual_last = test_truth[int(engine) - 1]
        trajectories.append(
            {
                "engine": int(engine),
                "points": [
                    {
                        "cycle": int(row.cycle),
                        "prediction": float(all_predictions[index]),
                        "actual": float(actual_last + frame["cycle"].max() - row.cycle),
                    }
                    for index, row in frame.iloc[::5].iterrows()
                ],
            }
        )
    return model, results, predictions, trajectories

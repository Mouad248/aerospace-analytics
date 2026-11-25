"""Small, transparent feature-drift monitor for replayed inference batches."""

from __future__ import annotations

import numpy as np


def build_reference_profile(matrix, feature_names, bins=10):
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names) or not len(values):
        raise ValueError("Reference features must be a nonempty aligned matrix")
    if not np.isfinite(values).all():
        raise ValueError("Reference features must be finite")
    profile = {"method": "population stability index", "bins": {}}
    for index, name in enumerate(feature_names):
        column = values[:, index]
        edges = np.unique(np.quantile(column, np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            edges = np.array([column[0] - 0.5, column[0] + 0.5])
        else:
            edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(column, bins=edges)
        profile["bins"][name] = {
            "edges": [float(x) for x in edges[1:-1]],
            "share": (counts / counts.sum()).tolist(),
        }
    return profile


def population_stability_index(reference_share, observed_share, epsilon=1e-6):
    reference = np.asarray(reference_share, dtype=float)
    observed = np.asarray(observed_share, dtype=float)
    if reference.shape != observed.shape or reference.ndim != 1 or not len(reference):
        raise ValueError("Reference and observed bins must have the same shape")
    if (
        not np.isfinite(reference).all()
        or not np.isfinite(observed).all()
        or np.any(reference < 0)
        or np.any(observed < 0)
        or reference.sum() <= 0
        or observed.sum() <= 0
    ):
        raise ValueError("Bin shares must be finite nonnegative values")
    expected = np.clip(reference / reference.sum(), epsilon, None)
    actual = np.clip(observed / observed.sum(), epsilon, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def assess_drift(reference_profile, matrix, feature_names, alert_threshold=0.2):
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(feature_names) or not len(values):
        raise ValueError("Observed features must be a nonempty aligned matrix")
    if not np.isfinite(values).all():
        raise ValueError("Observed features must be finite")
    scores = []
    for index, name in enumerate(feature_names):
        item = reference_profile["bins"].get(name)
        if item is None:
            continue
        counts, _ = np.histogram(values[:, index], bins=[-np.inf, *item["edges"], np.inf])
        score = population_stability_index(item["share"], counts)
        scores.append({"feature": name, "psi": score})
    scores.sort(key=lambda item: (-item["psi"], item["feature"]))
    maximum = scores[0]["psi"] if scores else 0.0
    return {
        "method": "population stability index",
        "alert_threshold": alert_threshold,
        "threshold_is_heuristic": True,
        "max_psi": maximum,
        "alert": maximum >= alert_threshold,
        "features_over_threshold": sum(x["psi"] >= alert_threshold for x in scores),
        "top_features": scores[:10],
    }

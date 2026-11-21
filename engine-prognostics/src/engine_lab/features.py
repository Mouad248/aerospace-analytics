"""Fit-only operating-regime normalization and strictly trailing features."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .data import SENSORS, SETTINGS


class FeatureBuilder:
    def __init__(self, regimes=1, window=20, seed=2026):
        if regimes not in (1, 6) or type(window) is not int or window < 2:
            raise ValueError("Use one or six regimes and a trailing window >=2")
        self.regimes, self.window, self.seed = regimes, window, seed

    def fit(self, frame):
        self.setting_scaler = StandardScaler().fit(frame[SETTINGS])
        settings = self.setting_scaler.transform(frame[SETTINGS])
        self.clusterer = KMeans(
            n_clusters=self.regimes, random_state=self.seed, n_init=10
        ).fit(settings)
        cluster = self.clusterer.predict(settings)
        raw = frame[SENSORS].to_numpy()
        self.active = np.std(raw, axis=0) > 1e-8
        raw = raw[:, self.active]
        floor = np.maximum(np.std(raw, axis=0) * 0.01, 1e-8)
        self.means, self.scales = [], []
        for regime in range(self.regimes):
            group = raw[cluster == regime]
            if not len(group):
                raise ValueError("Empty operating regime")
            self.means.append(group.mean(axis=0))
            self.scales.append(np.maximum(group.std(axis=0), floor))
        self.means, self.scales = np.array(self.means), np.array(self.scales)
        names = [n for n, active in zip(SENSORS, self.active, strict=True) if active]
        self.feature_names = [
            "cycle",
            *SETTINGS,
            *names,
            *[f"{n}_{stat}" for stat in ("mean", "std", "trend") for n in names],
        ]
        return self

    def transform(self, frame):
        settings = self.setting_scaler.transform(frame[SETTINGS])
        cluster = self.clusterer.predict(settings)
        raw = frame[SENSORS].to_numpy()[:, self.active]
        normalized = (raw - self.means[cluster]) / self.scales[cluster]
        means, stds, trends = (np.empty_like(normalized) for _ in range(3))
        for indices in frame.groupby("engine", sort=False).indices.values():
            x = normalized[indices]
            n = len(x)
            high = np.arange(1, n + 1)
            low = np.maximum(0, high - self.window)
            count = (high - low)[:, None]
            sums = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
            squares = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x * x, axis=0)])
            mean = (sums[high] - sums[low]) / count
            means[indices] = mean
            stds[indices] = np.sqrt(
                np.maximum(0, (squares[high] - squares[low]) / count - mean * mean)
            )
            trends[indices] = (x - x[low]) / np.maximum(1, count - 1)
        result = np.column_stack(
            [frame["cycle"].to_numpy(), settings, normalized, means, stds, trends]
        )
        if not np.isfinite(result).all():
            raise ValueError("Non-finite engineered features")
        return result

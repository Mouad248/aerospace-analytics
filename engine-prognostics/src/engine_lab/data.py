from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = [
    "engine",
    "cycle",
    *[f"setting_{i}" for i in range(1, 4)],
    *[f"sensor_{i}" for i in range(1, 22)],
]
SETTINGS, SENSORS = COLUMNS[2:5], COLUMNS[5:]
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def read_trajectory(source) -> pd.DataFrame:
    values = np.loadtxt(source, ndmin=2)
    if (
        values.ndim != 2
        or values.shape[1] != 26
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Trajectory requires 26 finite numeric columns")
    if not np.equal(values[:, :2], np.floor(values[:, :2])).all() or np.any(
        values[:, :2] < 1
    ):
        raise ValueError("Engine IDs and cycles must be positive integers")
    frame = pd.DataFrame(values, columns=COLUMNS)
    frame[["engine", "cycle"]] = frame[["engine", "cycle"]].astype(int)
    frame = frame.sort_values(["engine", "cycle"]).reset_index(drop=True)
    if frame.duplicated(["engine", "cycle"]).any():
        raise ValueError("Duplicate engine-cycle records")
    for _, group in frame.groupby("engine", sort=True):
        if not np.array_equal(group["cycle"].to_numpy(), np.arange(1, len(group) + 1)):
            raise ValueError(
                "Each trajectory must contain contiguous cycles starting at one"
            )
    return frame


def read_subset(archive: Path, subset: str):
    if subset not in SUBSETS:
        raise ValueError("Unknown C-MAPSS subset")
    with zipfile.ZipFile(archive) as bundle:
        members = {Path(n).name: n for n in bundle.namelist() if not n.endswith("/")}
        arrays = {}
        for name in (f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"):
            if name not in members:
                raise ValueError(f"Archive missing {name}")
            # ZipFile.read validates each member CRC; no filesystem extraction.
            arrays[name] = io.BytesIO(bundle.read(members[name]))
    train = read_trajectory(arrays[f"train_{subset}.txt"])
    test = read_trajectory(arrays[f"test_{subset}.txt"])
    rul = np.loadtxt(arrays[f"RUL_{subset}.txt"], ndmin=1)
    ids = sorted(test["engine"].unique())
    if (
        rul.ndim != 1
        or ids != list(range(1, len(ids) + 1))
        or len(rul) != len(ids)
        or not np.isfinite(rul).all()
        or np.any(rul < 0)
        or not np.equal(rul, np.floor(rul)).all()
    ):
        raise ValueError("Official RUL rows must align with contiguous test engine IDs")
    return train, test, rul


def split_engines(frame: pd.DataFrame, seed=2026):
    ids = list(map(int, frame["engine"].unique()))
    if len(ids) < 15:
        raise ValueError("At least 15 run-to-failure engines are required")
    ids.sort(
        key=lambda unit: hashlib.sha256(f"engine-lab:{seed}:{unit}".encode()).digest()
    )
    size = max(5, int(len(ids) * 0.2))
    return {
        "fit": sorted(ids[2 * size :]),
        "validation": sorted(ids[:size]),
        "calibration": sorted(ids[size : 2 * size]),
    }


def training_rul(frame):
    return (
        frame.groupby("engine")["cycle"].transform("max") - frame["cycle"]
    ).to_numpy(dtype=float)


def pseudo_endpoints(frame: pd.DataFrame, seed=2026):
    """One reproducible pre-failure cutoff per engine, using only its prefix."""
    indices = []
    for engine, group in frame.groupby("engine", sort=True):
        digest = hashlib.sha256(f"cutoff:{seed}:{engine}".encode()).digest()
        fraction = 0.35 + 0.5 * int.from_bytes(digest[:8], "big") / (2**64 - 1)
        count = max(1, min(len(group) - 1, int(len(group) * fraction)))
        indices.append(int(group.index[count - 1]))
    return np.array(indices)


def last_indices(frame):
    return frame.groupby("engine", sort=True).tail(1).index.to_numpy()

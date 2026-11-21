"""Independent original-label reconciliation, metrics and fit-only statistics."""

import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    output = ROOT / "reports/latest"
    data = json.loads((output / "results.json").read_text(encoding="utf-8"))
    archive = ROOT / "data/raw/CMAPSSData.zip"
    assert (
        hashlib.sha256(archive.read_bytes()).hexdigest()
        == data["provenance"]["archive_sha256"]
    )
    total = 0
    with zipfile.ZipFile(archive) as bundle:
        members = {Path(n).name: n for n in bundle.namelist()}
        for subset, record in data["subsets"].items():
            m, pred = record["metrics"], record["predictions"]
            train = np.loadtxt(io.BytesIO(bundle.read(members[f"train_{subset}.txt"])))
            test = np.loadtxt(io.BytesIO(bundle.read(members[f"test_{subset}.txt"])))
            truth = np.loadtxt(io.BytesIO(bundle.read(members[f"RUL_{subset}.txt"])))
            assert len(pred) == len(truth) == m["test_engines"]
            assert len(train) == m["train_rows"] and len(test) == m["test_rows"]
            fit, validation, calibration = (
                set(m["split"][key]) for key in ("fit", "validation", "calibration")
            )
            assert (
                not fit & validation
                and not fit & calibration
                and not validation & calibration
            )
            assert fit | validation | calibration == set(train[:, 0].astype(int))
            ranked = sorted(
                fit | validation | calibration,
                key=lambda unit: hashlib.sha256(
                    f"engine-lab:{m['seed']}:{unit}".encode()
                ).digest(),
            )
            assert sorted(ranked[: len(validation)]) == sorted(validation)
            expected_selected = min(
                m["validation"], key=lambda name: (m["validation"][name]["mae"], name)
            )
            assert m["selected_model"] == expected_selected
            errors = []
            coverage = width = tp = fp = fn = tn = 0
            for i, p in enumerate(pred):
                assert p["engine"] == i + 1 and p["rul_actual"] == truth[i]
                assert p["cycle"] == int(test[test[:, 0] == i + 1, 1].max())
                assert 0 <= p["rul_predicted"] <= m["training_rul_cap"]
                assert (
                    abs(
                        p["lower"]
                        - max(0, p["rul_predicted"] - m["interval"]["radius"])
                    )
                    < 1e-10
                )
                assert (
                    abs(p["upper"] - p["rul_predicted"] - m["interval"]["radius"])
                    < 1e-10
                )
                errors.append(p["rul_predicted"] - truth[i])
                coverage += p["lower"] <= truth[i] <= p["upper"]
                width += p["upper"] - p["lower"]
                flagged, positive = p["lower"] <= 30, truth[i] <= 30
                tp += flagged and positive
                fp += flagged and not positive
                fn += not flagged and positive
                tn += not flagged and not positive
            assert math.isclose(
                m["test"]["mae"], sum(map(abs, errors)) / len(errors), abs_tol=1e-9
            )
            assert math.isclose(
                m["test"]["rmse"],
                math.sqrt(sum(e * e for e in errors) / len(errors)),
                abs_tol=1e-9,
            )
            score = sum(math.expm1(-e / 13 if e < 0 else e / 10) for e in errors)
            assert math.isclose(m["test"]["asymmetric_score"], score, rel_tol=1e-10)
            assert math.isclose(
                m["interval"]["empirical_test_coverage"],
                coverage / len(errors),
                abs_tol=1e-10,
            )
            assert math.isclose(
                m["interval"]["mean_width"], width / len(errors), abs_tol=1e-9
            )
            assert (tp, fp, fn, tn) == tuple(
                m["review_screen"][k]
                for k in (
                    "true_positive",
                    "false_positive",
                    "false_negative",
                    "true_negative",
                )
            )
            model_path = ROOT / "models" / f"{subset}.joblib"
            assert (
                hashlib.sha256(model_path.read_bytes()).hexdigest() == m["model_sha256"]
            )
            model = joblib.load(model_path)
            fit_rows = train[np.isin(train[:, 0], list(fit))]
            # Confirm actual saved normalization used fitting engines only.
            np.testing.assert_allclose(
                model.features.setting_scaler.mean_,
                fit_rows[:, 2:5].mean(axis=0),
                atol=1e-9,
            )
            scaled = (
                fit_rows[:, 2:5] - model.features.setting_scaler.mean_
            ) / model.features.setting_scaler.scale_
            with threadpool_limits(limits=2):
                clusters = model.features.clusterer.predict(scaled)
            sensors = fit_rows[:, 5:][:, model.features.active]
            for regime in range(m["regimes"]):
                np.testing.assert_allclose(
                    model.features.means[regime],
                    sensors[clusters == regime].mean(axis=0),
                    atol=1e-8,
                )
            total += len(pred)
    with (output / "predictions.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == total
    for row in rows:
        saved = data["subsets"][row["subset"]]["predictions"][int(row["engine"]) - 1]
        assert all(float(row[k]) == saved[k] for k in saved)
    result = {
        "status": "passed",
        "subsets": len(data["subsets"]),
        "official_test_engines": total,
        "metrics_recomputed_from_original_labels": True,
        "fit_only_normalization_verified": True,
        "engine_splits_disjoint_and_reproduced": True,
        "model_hashes_verified": True,
        "csv_rows_verified": len(rows),
    }
    (output / "independent_verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

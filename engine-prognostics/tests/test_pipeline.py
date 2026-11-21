import io
import json
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from engine_lab.cli import main
from engine_lab.data import (
    COLUMNS,
    read_subset,
    read_trajectory,
    split_engines,
    training_rul,
)
from engine_lab.features import FeatureBuilder
from engine_lab.modelling import conformal_radius, metrics, review_metrics, train_subset
from engine_lab.report import write_report


def fleet(engines=20, cycles=60):
    rows = []
    for unit in range(1, engines + 1):
        for cycle in range(1, cycles + 1):
            rows.append(
                [
                    unit,
                    cycle,
                    0.0,
                    0.0,
                    100.0,
                    *[
                        20 + i + 0.01 * cycle * (i + 1) + 0.001 * unit
                        for i in range(21)
                    ],
                ]
            )
    result = pd.DataFrame(rows, columns=COLUMNS)
    result[["engine", "cycle"]] = result[["engine", "cycle"]].astype(int)
    return result


def stream(frame):
    value = io.StringIO()
    np.savetxt(value, frame.to_numpy())
    value.seek(0)
    return value


def test_parser_sorts_and_preserves_values():
    frame = fleet(2, 4)
    got = read_trajectory(stream(frame.iloc[::-1]))
    np.testing.assert_allclose(got, frame)


@pytest.mark.parametrize(
    "fault", ["duplicate", "gap", "fractional_id", "nan", "columns"]
)
def test_bad_trajectories_rejected(fault):
    frame = fleet(2, 4)
    if fault == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif fault == "gap":
        frame = frame.drop(index=1)
    elif fault == "fractional_id":
        frame = frame.astype(float)
        frame.loc[0, "engine"] = 1.5
    elif fault == "nan":
        frame.loc[0, "sensor_1"] = np.nan
    else:
        frame = frame.drop(columns="sensor_21")
    with pytest.raises(ValueError):
        read_trajectory(stream(frame))


def test_rul_and_split_engine_boundaries():
    frame = fleet()
    rul = training_rul(frame)
    assert list(rul[:3]) == [59, 58, 57] and rul[59] == 0 and rul[60] == 59
    split = split_engines(frame)
    assert {k: len(v) for k, v in split.items()} == {
        "fit": 10,
        "validation": 5,
        "calibration": 5,
    }
    assert split_engines(frame.iloc[::-1]) == split
    assert len({unit for group in split.values() for unit in group}) == 20
    with pytest.raises(ValueError):
        split_engines(fleet(3))


def test_trailing_features_have_no_future_or_other_engine_leakage():
    frame = fleet(2, 40)
    with threadpool_limits(limits=2):
        builder = FeatureBuilder(window=5).fit(frame)
        original = builder.transform(frame)
        changed = frame.copy()
        changed.loc[(changed.engine == 1) & (changed.cycle > 10), "sensor_1"] += 500
        alternative = builder.transform(changed)
        np.testing.assert_allclose(original[:10], alternative[:10])
        prefix = builder.transform(frame.iloc[:10].copy())
        np.testing.assert_allclose(original[:10], prefix)
        # First row of each engine has one value, zero trailing standard deviation.
        std_index = builder.feature_names.index("sensor_1_std")
        assert original[0, std_index] == pytest.approx(0, abs=1e-7)
        assert original[40, std_index] == pytest.approx(0, abs=1e-7)
        mean_index = builder.feature_names.index("sensor_1_mean")
        current_index = builder.feature_names.index("sensor_1")
        assert original[40, mean_index] == pytest.approx(original[40, current_index])
        assert original[8, mean_index] == pytest.approx(
            original[4:9, current_index].mean()
        )


def test_split_conformal_finite_sample_rank_and_bad_calibration():
    assert conformal_radius(np.arange(1, 21), 0.2) == 17
    for values, alpha in [
        ([], 0.2),
        ([1], 0.01),
        ([-1, 2], 0.5),
        ([float("nan")], 0.5),
        ([1, 2], 0),
    ]:
        with pytest.raises(ValueError):
            conformal_radius(values, alpha)


def test_metrics_hand_calculation_and_asymmetric_penalty():
    actual = np.array([20.0, 40.0])
    result = metrics(actual, np.array([10.0, 50.0]))
    assert result["mae"] == 10 and result["rmse"] == 10 and result["mean_error"] == 0
    assert result["asymmetric_score"] == pytest.approx(np.expm1(10 / 13) + np.expm1(1))
    assert (
        metrics(np.array([20]), np.array([30]))["asymmetric_score"]
        > metrics(np.array([20]), np.array([10]))["asymmetric_score"]
    )
    with pytest.raises(ValueError):
        metrics(np.array([1, 2]), np.array([1]))


def test_review_denominators_include_undefined_cases():
    result = review_metrics([10, 20, 80, 90], [5, 40, 5, 50], 30)
    assert (
        result["true_positive"],
        result["false_positive"],
        result["false_negative"],
        result["true_negative"],
    ) == (1, 1, 1, 1)
    assert result["precision"] == 0.5 and result["recall"] == 0.5
    assert review_metrics([80], [50], 30)["recall"] is None


def test_archive_rul_alignment_and_rejection(tmp_path):
    path = tmp_path / "data.zip"

    def save(rul):
        with zipfile.ZipFile(path, "w") as z:
            for kind in ("train", "test"):
                z.writestr(f"folder/{kind}_FD001.txt", stream(fleet(2, 4)).getvalue())
            z.writestr("folder/RUL_FD001.txt", rul)

    save("10\n20\n")
    train, test, rul = read_subset(path, "FD001")
    assert len(train) == 8 and len(test) == 8 and list(rul) == [10, 20]
    save("10\n")
    with pytest.raises(ValueError):
        read_subset(path, "FD001")
    save("10.5\n20\n")
    with pytest.raises(ValueError):
        read_subset(path, "FD001")


def test_real_pipeline_baseline_selection_roundtrip_and_label_free_inference(tmp_path):
    train = fleet(20, 60)
    test = fleet(3, 30)
    with threadpool_limits(limits=2):
        model, result, _predictions, _ = train_subset(
            train, test, np.array([30.0, 30.0, 30.0]), "FD001"
        )
    assert result["selected_model"] == "Age baseline" and result["test"]["rmse"] == 0
    assert result["interval"]["radius"] == 0
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_array_equal(loaded.terminal(test)[0], [30, 30, 30])
    input_path = tmp_path / "observed.txt"
    input_path.write_text(stream(test).getvalue(), encoding="utf-8")
    output = tmp_path / "inference.csv"
    main(
        [
            "predict",
            "--model",
            str(path),
            "--input",
            str(input_path),
            "--output",
            str(output),
        ]
    )
    assert len(output.read_text().splitlines()) == 4
    assert "rul_actual" not in output.read_text()


def test_report_payload_cannot_close_script(tmp_path):
    value = {"unsafe": "</script><img onerror=bad>"}
    path = tmp_path / "r.html"
    write_report(value, path)
    text = path.read_text(encoding="utf-8")
    assert "</script><img" not in text
    assert (
        json.loads(
            text.split('<script type="application/json" id="data">')[1].split(
                "</script>"
            )[0]
        )
        == value
    )

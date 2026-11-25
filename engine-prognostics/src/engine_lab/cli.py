from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits

from .data import SUBSETS, read_subset, read_trajectory
from .modelling import train_subset
from .mlops import promotion_gate
from .report import write_report

ARCHIVE_URL = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
ARCHIVE_SHA = "74bef434a34db25c7bf72e668ea4cd52afe5f2cf8e44367c55a82bfd91a5a34f"


def dump(path, obj):
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value)}")

    path.write_text(
        json.dumps(obj, indent=2, default=convert, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def retrieve(target: Path):
    if target.exists():
        if hashlib.sha256(target.read_bytes()).hexdigest() != ARCHIVE_SHA:
            raise ValueError(
                "Existing archive differs from pinned source; it was preserved"
            )
        print("Existing NASA archive verified")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ARCHIVE_URL, timeout=90) as response:
        content = response.read(50_000_001)
    if len(content) > 50_000_000 or hashlib.sha256(content).hexdigest() != ARCHIVE_SHA:
        raise ValueError("NASA archive checksum or size mismatch")
    temporary = target.with_suffix(".part")
    temporary.write_bytes(content)
    os.replace(temporary, target)


def train(args):
    output, models = args.output.resolve(), args.models.resolve()
    if args.archive.resolve().is_relative_to(output) or output == models:
        raise ValueError("Use separate raw, report and model paths")
    output.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    results = {
        "provenance": {
            "generated_utc": datetime.now(UTC).isoformat(),
            "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "data_kind": "simulated",
            "test_labels_used_for_training": False,
            "model_links": {},
        },
        "subsets": {},
    }
    all_rows = []
    if not args.no_tracking:
        import mlflow
        from .mlops import log_training_run

        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment("Engine Prognostics | C-MAPSS")
    for subset in args.subsets:
        print(
            f"Training {subset}: fit / validation / calibration engines kept separate",
            flush=True,
        )
        train_data, test_data, truth = read_subset(args.archive, subset)
        with threadpool_limits(limits=2):
            model, metrics, predictions, trajectories = train_subset(
                train_data, test_data, truth, subset, args.seed, args.rul_cap
            )
        destination = models / f"{subset}.joblib"
        joblib.dump(model, destination, compress=3)
        restored = joblib.load(destination)
        check = restored.terminal(test_data)[0]
        if not np.allclose(
            check, [p["rul_predicted"] for p in predictions], atol=1e-10, rtol=0
        ):
            raise ValueError("Saved model failed prediction round-trip")
        metrics["model_reload_verified"] = True
        metrics["model_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        gate = promotion_gate(metrics)
        metrics["promotion_gate"] = gate
        results["provenance"]["model_links"][subset] = Path(
            os.path.relpath(destination, output)
        ).as_posix()
        for name in ("train_engines", "test_engines"):
            metrics[name] = int(metrics[name])
        results["subsets"][subset] = {
            "metrics": metrics,
            "predictions": predictions,
            "trajectories": trajectories,
        }
        if not args.no_tracking:
            log_training_run(
                subset=subset,
                model=model,
                model_path=destination,
                metrics=metrics,
                seed=args.seed,
                rul_cap=args.rul_cap,
                archive_sha256=results["provenance"]["archive_sha256"],
            )
        all_rows.extend({"subset": subset, **row} for row in predictions)
        print(
            f"{subset}: selected {metrics['selected_model']}; official RMSE {metrics['test']['rmse']:.2f}, "
            f"coverage {metrics['interval']['empirical_test_coverage']:.1%}",
            flush=True,
        )
    dump(output / "results.json", results)
    dump(
        models / "model_cards.json",
        {k: v["metrics"] for k, v in results["subsets"].items()},
    )
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    # Normalize numpy scalar counts before embedding the offline payload.
    normalized = json.loads((output / "results.json").read_text(encoding="utf-8"))
    write_report(normalized, output / "index.html")
    print(f"Report: {output / 'index.html'}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="C-MAPSS remaining-life research pipeline"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    dl = commands.add_parser("download")
    dl.add_argument("--archive", type=Path, default=Path("data/raw/CMAPSSData.zip"))
    tr = commands.add_parser("train")
    tr.add_argument("--archive", type=Path, default=Path("data/raw/CMAPSSData.zip"))
    tr.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    tr.add_argument("--output", type=Path, default=Path("reports/latest"))
    tr.add_argument("--models", type=Path, default=Path("models"))
    tr.add_argument("--seed", type=int, default=2026)
    tr.add_argument("--rul-cap", type=float, default=125)
    tr.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        help="MLflow tracking URI; defaults to a local SQLite store",
    )
    tr.add_argument(
        "--no-tracking",
        action="store_true",
        help="retrain and export models without opening an MLflow run",
    )
    pred = commands.add_parser(
        "predict",
        help="predict terminal RUL from a trusted fitted model and observed prefixes",
    )
    pred.add_argument("--model", type=Path, required=True)
    pred.add_argument("--input", type=Path, required=True)
    pred.add_argument("--output", type=Path, required=True)
    serve = commands.add_parser("serve", help="run the prediction and monitoring API")
    serve.add_argument("--model", type=Path, required=True)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.command == "download":
        retrieve(args.archive)
    elif args.command == "train":
        train(args)
    elif args.command == "predict":
        model = joblib.load(args.model)
        frame = read_trajectory(args.input)
        prediction, lower, upper = model.terminal(frame)
        terminal = frame.groupby("engine", sort=True).tail(1)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["engine", "cycle", "rul_predicted", "lower", "upper"])
            for i, row in enumerate(terminal.itertuples()):
                writer.writerow(
                    [row.engine, row.cycle, prediction[i], lower[i], upper[i]]
                )
        print(f"Predicted {len(terminal)} engines without loading any RUL label file")
    else:
        import uvicorn
        from .api import create_app

        uvicorn.run(
            create_app(args.model.resolve()),
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=True,
        )

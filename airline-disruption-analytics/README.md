# Airline Disruption Forecasting and Recovery Scenarios

Use actual U.S. flight records to forecast disruption risk and arrival-delay severity from information available before departure, trace aircraft-turn sequences on a held-out day, and allocate a constrained buffer budget in an explicit what-if scenario.

## What it demonstrates

- **Large-scale data preparation:** streamed PyArrow-to-Parquet preparation, then DuckDB rolling-window features over six months of BTS flight records.
- **Supervised ML:** LightGBM disruption classifier and 90th-percentile arrival-delay regressor with a May-only finite-sample conformal correction.
- **Leakage control:** chronological train (January-April 2025), validation/calibration (May), and held-out test (June); lagged 28-day history features exclude the current date.
- **Model evaluation:** May-fitted isotonic probability calibration, June ranking/Brier/quantile coverage/error; compare disruption forecasts against a lagged-history baseline.
- **Aircraft rotations:** NetworkX connects same-tail flights only where airport continuity, actual UTC timing and turnaround windows are plausible. Tail IDs are pseudonymized in public artifacts.
- **Decision support:** OR-Tools CP-SAT allocates a stated, hypothetical schedule-buffer budget across high-risk flights, maximizing disruption-probability-weighted q90 buffer minutes. The weights and constraints are scenario assumptions, not observed airline policy.

**Prediction target:** a flight is disrupted if it is cancelled, diverted or arrives at least 15 minutes late. Delay quantiles use completed, non-diverted flights. The pre-departure model excludes actual departure/arrival times, delay causes and cancellation reasons.

## Run

From this folder in PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
.venv/Scripts/python.exe scripts/download_data.py
.venv/Scripts/python.exe scripts/prepare_data.py
.venv/Scripts/python.exe run.py train
.venv/Scripts/python.exe run.py scenario
.venv/Scripts/python.exe scripts/write_report.py
.venv/Scripts/python.exe run.py serve
```

Open http://127.0.0.1:8060/reports/latest/. Raw monthly archives and prepared Parquet are excluded from Git. File hashes, months and selected fields are in `data/manifest.json`.

## Scope and interpretation

The source is the U.S. Bureau of Transportation Statistics Marketing Carrier On-Time Performance table, using six months in 2025. It records scheduled and actual flight performance and aircraft tail numbers. This is U.S. data, not a Moroccan airline sample. Airport codes and carrier codes identify public service locations/operators; aircraft tail numbers are hashed before they enter the prepared modeling data.

The history does not contain passenger itineraries, crew legality, gates, weather forecasts, aircraft maintenance state, airline costs or real recovery decisions. The graph is a reconstructed sample of observed tail continuity, and the CP-SAT exercise uses assumed buffer costs/capacity. It illustrates prediction-to-optimization workflow; it is not an airline dispatch system or a measured recovery benefit.

See [data and method](docs/DATA_AND_METHOD.md) and [held-out findings](docs/FINDINGS.md).

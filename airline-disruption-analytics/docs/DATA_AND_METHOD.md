# Data and method

## Source and sample

The U.S. BTS Marketing Carrier On-Time Performance dataset contains monthly non-stop domestic flight records, scheduled and actual departure/arrival times, cancellations/diversions, delay causes, distance and aircraft tail numbers. This project downloads January-June 2025. January-April form the fitting period, May is used for calibration/selection, and June is reserved for final temporal evaluation. Source URLs and SHA256 values are recorded in `data/manifest.json`; raw archives are ignored by Git. The BTS page describes the fields and release cadence; attribution is retained.

The preprocessing step streams the CSV member from each ZIP into Parquet, selecting only fields needed for the stated analysis. Duplicate-flagged rows are excluded. Tail numbers are hashed immediately and source airline/operator codes remain coded identifiers. No passenger names, ticket records or personal identifiers are present in the selected fields.

## Prediction design

The decision point is pre-departure. The classifier target is `disrupted = cancelled OR diverted OR arrival delay >= 15 minutes`. The delay model predicts the 90th percentile of `ArrDelay` for completed, non-diverted flights. A finite-sample additive split-conformal correction is fitted from May residuals and frozen before June scoring. Features are scheduled carrier/route/calendar/time/distance plus 28-calendar-day origin, destination, route and carrier disruption rates. DuckDB window frames end before the flight date, so same-day outcomes cannot leak into the historical rates.

Do not use actual departure time/delay, actual arrival, cancellation reason, cause-of-delay minutes or future observations as pre-departure features. These source columns are retained only for labels and post-hoc descriptive analyses. LightGBM handles the moderate-cardinality categorical route/airport/operator features. The May probability output is calibrated with isotonic regression; June remains untouched for final reporting. Airport, destination and carrier history features average prior daily route-cell disruption rates; the route baseline averages that route's prior daily rates. The data do not contain weather forecasts, so the model is an operational-history baseline rather than a weather-aware forecast.

## Aircraft graph and optimizer

For a reproducible held-out day, local actual departure times are converted using airport IANA time zones; observed block duration estimates actual arrival in UTC. A NetworkX directed graph links consecutive records with a pseudonymized aircraft tail only if destination/origin continuity matches and actual turn time falls in the declared plausibility window. The reported association is between preceding arrival delay and the next flight's arrival delay; it is descriptive, not a causal estimate.

The CP-SAT scenario selects per-flight buffers from 0-30 minutes in five-minute steps under a fixed total buffer budget. It maximizes the assumed disruption-probability-weighted q90 minutes buffered. The resulting residual q90 is a transparent what-if calculation. Weights and resource caps are authored assumptions. This is a small, reproducible allocation demonstration; the data lack airline-specific schedule-change costs, passenger reaccommodation, crews, gates and maintenance constraints.

## Validation and boundaries

The train/validation/test chronology prevents later calendar outcomes from fitting earlier rows. Lagged history for later months may use earlier observed days, as it would in an operational rolling process. The May conformal correction targets 90% marginal coverage under exchangeability; June is a later month and may have a shifted delay distribution, so coverage is measured again rather than assumed. A six-month slice is not a full-year seasonal analysis, and June is one holdout month. Carrier reporting, exceptional weather and schedule changes can shift distributions. U.S. public records do not establish performance for Morocco-based carriers.

The project predicts disruption indicators and delay distributions for analysis. It does not promise real-time accuracy, causality, safety, passenger-impact reduction, airline cost savings or production deployment. The optimizer is a scenario model, not a validated recovery recommendation.

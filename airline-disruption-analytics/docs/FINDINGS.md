# Held-out findings

June 2025 temporal holdout: 674,179 labeled flights; disruption prevalence 29.9%.

## Disruption model

- Isotonic-calibrated LightGBM ROC AUC: 0.6549; average precision: 0.4227.
- Brier score: 0.19952; preceding-28-day route-rate baseline: 0.20473.
The baseline comparison matters: a more complex model is useful only if it improves on simple recent route history. Isotonic calibration was fitted on May only; June was held out.

## Arrival-delay quantile

The LightGBM q90 prediction received a May-only conformal correction of 14.70 minutes. On 659,673 completed, non-diverted June flights, coverage moved from 81.8% to 87.1%; calibrated MAE is 54.60 minutes (raw: 44.39). May coverage after correction was 90.0%. June remains a temporal-shift check, not a guarantee of 90% future coverage.

## Network and buffer scenario

On 2025-06-30, reconstructed 16,242 plausible consecutive-flight links from pseudonymized aircraft identifiers. The descriptive correlation between preceding arrival delay and next-flight arrival delay was 0.48878307543151894.
The hypothetical 240-minute buffer scenario allocated 240 minutes across 30 high-q90 flights. Mean modeled q90 was 227.6 minutes before and 219.6 after allocation, under the authored scenario objective. This is not a measured operational intervention.

## Scope limits

- Only pre-departure schedule/operator/location/distance and prior observed 28-day history are model inputs.
- Six months of U.S. records; June is one temporal holdout month, and results are not transferable without validation.
- No weather forecast, passenger connections, crew, gate, maintenance or airline-cost data.
- CP-SAT buffer allocation is an assumption-led scenario, not measured airline recovery benefit.

See [full metrics](../reports/latest/results.json), [data and method](DATA_AND_METHOD.md), and [run instructions](../README.md).

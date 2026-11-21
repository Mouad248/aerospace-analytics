# Verification — 18 November 2026

Implemented and evaluated on all four original C-MAPSS subsets, with Python 3.13, scikit-learn 1.9.1, NumPy 2.5.5 and pandas 3.1.1.

| Check | Result |
|---|---|
| Automated tests | **14 passed** |
| Ruff formatting/static checks | Passed |
| Original archive | SHA256 identity and per-member ZIP CRC verification |
| Scope | FD001, FD002, FD003 and FD004; 709 training / 707 official test units |
| Source rows | 160,359 training / 104,897 test rows |
| Engine splits | Disjoint fitting/validation/calibration sets, reproduced independently |
| Saved normalization | Actual artifact settings/sensor statistics checked against fitting engines only |
| Official metrics | MAE, RMSE, asymmetric score, interval coverage/width and review counts independently recomputed from publisher labels |
| Persisted models | Four joblib artifacts reloaded; hashes and terminal predictions verified |
| Source-to-CSV reconciliation | **707 terminal predictions** matched |
| Browser | All four fleets, trajectory selector, review horizon, model links and all 707 CSV download rows checked |
| Responsive/offline execution | Desktop 1440×1050 and mobile 390×844; no document overflow, JavaScript errors or external requests |
| Independent installation | Fresh wheel environment, inference outside source checkout; all **707 predictions and intervals** reproduced without loading target files |

Evidence: [independent verification](independent_verification.json), [browser verification](browser_verification.json), [installation verification](package_verification.json), [full metrics](results.json), [model cards](../../models/model_cards.json).

The synthetic tests exercise malformed trajectories/targets, unit boundaries, future-sensor mutation, exact trailing means, finite-sample calibration ranks, hand-calculated asymmetric penalties, zero-denominator review metrics, known baseline selection, serialization and inference without any label file. Real benchmark evaluation and independent original-label reconciliation are separate from those synthetic tests.

Selection is driven by validation engines, not the official test result. Gradient boosting was selected for FD001/2/3; Ridge for FD004. Point targets are capped at 125 cycles during training, while the main official-test metrics use uncapped RUL. All four empirical coverage rates and interval widths are retained, including undercoverage on FD002 and overcoverage on FD003.

All observations are simulated. Nominal conformal coverage assumes exchangeability; real-world censoring, operating changes and safety certification are outside this benchmark validation. No intervention effect or maintenance cost saving is inferred.

# Reproduction verification

Validated on 2026-10-30 in a fresh Python 3.10.9 environment on Windows. Dependency installation and `pip check` passed. The installed version inventory is environment-validated-cpu.txt; use the README commands for CPU PyTorch installation rather than assuming a CUDA wheel is available.

| Check | Evidence | Status |
|---|---|---|
| Full-precision scores | 360 MAE/RMSE/NSE/HSS/accuracy/count comparisons with original run JSON; maximum absolute difference 4.44e-16, required tolerance 1e-10 | Passed |
| Published numerical tables | Compare the recomputed values at the original four-decimal precision | Passed |
| MAE tests and lead-time intervals | Recompute Wilcoxon/Holm results and 2,000-resample, 14-calendar-day block bootstrap | Passed |
| Threshold sensitivity | Four thresholds, descriptive summaries, ranks and original unadjusted one-sided HSS p-values | Passed; supplementary p-values are not the main MAE test |
| Selected Fig.6 windows | NGR6/8, T-PGNN5/8, HAPP2/8, PM-GP0/8 correct, only for the eight selected windows | Passed |
| Daily-data reconstruction | Ten raw station subsets → QC/daily values/origin flags; values within 1e-10 | Passed |
| Four-method CPU smoke | Chaohu, held-out 2023 block, reduced budgets; 80 origins × 7 days for each algorithm | Passed |
| Full 40-run training replication | Frozen training entry provided; not rerun in this release task | Not verified |
| Cross-platform training equality | GPU/OS/library differences not tested as exact replication | Not verified |

The numerical result figures, QC plots and coordinate map were regenerated. Display/font differences are not interpreted as different predictions. The methods schematic, original cartographic SVG and graphical abstract are explanatory assets; copying them is not numerical reproduction.

Original T-PGNN deliberately requires CUDA by default. Smoke mode explicitly selects its existing CPU test switch, leaving frozen algorithm bytes unchanged; formal training keeps the default. Before this correction an initial CPU smoke attempt failed early, and no failed trial was counted as a passing result. A new complete four-method smoke run then passed.

Original run-log version strings are retained verbatim (without machine names) and differ from the separately validated release environment. This report does not claim all original historical environments were recreated. Actual parameter summaries and selected configurations are provided; complete pretrained checkpoints were not retained.

Private absolute paths, personal hostnames, paper drafts, annotations, credentials, resume caches and Optuna databases are excluded. The repository owner confirmed data redistribution permission; no written permission URL was supplied or invented. See DATA_LICENSE.md.

`reference/verification_local.json` is the machine-readable local evidence. Publication download verification is recorded separately after the fixed release is fetched from GitHub; absence of that record is not proof of a completed external check.

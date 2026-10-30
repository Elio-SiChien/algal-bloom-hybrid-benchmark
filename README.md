# Algal bloom hybrid benchmark

Reproducibility materials for **Where process knowledge should enter machine learning for 7-day algal bloom forecasting** (manuscript under review/preparation; no published paper DOI is claimed).

Compare four published-framework adaptations on a common seven-day algal-density task in ten lakes and reservoirs: **residual correction (PM-GP), equation embedding (NGR / Inside), process regularization (T-PGNN / Alongside), and potential-based mapping (HAPP)**. Persistence and monthly climatology are the two baselines.

## Quick start: reproduce the paper's numerical results

Python **3.10** is the validated target. Create a new environment; no GPU or private account is needed for results mode.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python reproduce.py --mode results
```

The program downloads **the fixed v1.0.0-review prediction asset**, verifies the ZIP and every member, and recomputes scores from forecasts and genuine observations. It does not copy the reference score table as its result. An optional offline input is supported:

```bash
python reproduce.py --mode results --predictions /path/to/formal-predictions-v3.zip
```

Outputs appear in a new `outputs/results_<timestamp>/` directory. Inspect `verification.json`, `results/metrics_full_precision.csv`, tables, lead-time intervals, case-window comparisons, PNG/PDF result figures, and `results/si_data_processing/` for QC plots. Missing inputs, wrong hashes, incomplete methods or numerical mismatches stop the run. No synthetic fallback is used. PNG/PDF bytes can differ by fonts/platform; underlying values are the comparison target.

## Four-method smoke test and full training

```bash
python -m pip install -r requirements-training.txt
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
python reproduce.py --mode smoke
```

The smoke test runs all four algorithms on Chaohu at reduced budgets, using a held-out block of **2023 training data**, not the 2024 formal evaluation year. It verifies fitting, seven-day trajectories and scoring. It is not a formal replication or a performance comparison. Smoke mode explicitly sets the frozen implementation’s M1_DEVICE=cpu test switch. Formal T-PGNN defaults to NVIDIA CUDA and is checked before a full run begins; complete training is substantially more expensive.

```bash
python reproduce.py --mode train --method all --site all
python reproduce.py --mode train --method NGR --site chaohu
```

Train mode uses the frozen formal budgets and five-seed aggregation where applicable. It includes both baselines. It never overwrites the reference predictions or author's experimental archive. For formal T-PGNN training, install an appropriate CUDA PyTorch build instead of the CPU wheel; CUDA is not needed for results mode or the other three methods. Hardware/library differences can affect stochastic training. This release supplies saved predictions and parameter summaries, **not four complete pretrained model checkpoints**.

## Rebuild the selected-station daily data

```bash
python reproduce.py --mode prepare
python -m unittest discover -s tests -v
```

Prepare mode rebuilds QC, daily aggregation, interpolation and origins from the ten published station subsets, then compares daily values (absolute tolerance `1e-10`) and origin flags to the frozen inputs. The original national archive and this ten-station subset have different hashes. The 176-station screening census is provided for inspection; the full national source archive is not redistributed, so the census is not claimed to be independently regenerated here.

## Scientific conditions to retain

- Train/configure on 2023; frozen evaluation on 2024, separately calibrated for every waterbody. This is not zero-shot transfer.
- Predictions cover target days 1–7. Direction uses the Theil–Sen ln-density slope over the same observed target dates, with a prespecified ±0.02 day−1 stable-class tolerance and at least four genuine target observations.
- Interpolation may support training/input continuity; interpolated evaluation targets are excluded. Missing-data masks and training/CV buffers are frozen.
- Future forcing is **ERA5-Land reanalysis**, not operational weather forecasts. Nutrients are held at the origin where applicable; future water temperature is estimated from a training-only regression.
- Tuning budgets differ by algorithm. Configuration selection used validation MAE even though direction judgment is the primary paper evaluation.
- PM-GP corrects daily growth-rate residuals before state integration and resets to the observed state at each origin. NGR jointly learns its growth network and two loss parameters. HAPP maps seven-day-lagged environmental potential, without observed algal state at the origin.
- Four concrete implementations are compared; their differences do not isolate the causal effect of coupling position. Positive HSS is neither a significance test nor demonstrated management benefit.

See [data dictionary and provenance](docs/DATA.md), [model sources and adaptations](docs/METHODS.md), [reproduction verification](docs/VERIFICATION.md), and [data terms](DATA_LICENSE.md). Exact original version strings are retained in `reference/original_run_environments.json`; the validated release environment is recorded separately. No original environment is silently reconstructed from incomplete logs.

## What is reproduced

`results` recomputes MAE/RMSE/NSE/HSS, direction identification rates, across-waterbody summaries, Wilcoxon/Holm MAE comparisons, threshold descriptive summaries, eight example windows, and calendar-block bootstrap lead-time intervals. Full-precision metrics are checked against stored run JSON with absolute tolerance `1e-10`; published four-decimal CSV values are checked at their stated precision. Threshold sensitivity also reproduces the stored unadjusted, one-sided waterbody-level HSS p-values; these are supplementary and are not the main MAE test.

Numerical figures are regenerated from data. Their display layout may differ from the manuscript's final typography. The study-map SVG, method schematic and graphical abstract are explanatory assets, not numerical evidence or training outputs. Table S1 physical characteristics retain source-specific conditions; unresolved mean depths remain missing.

## Reuse and citation

Original benchmark code is MIT licensed. Third-party data and cartography retain their own terms; see DATA_LICENSE.md and THIRD_PARTY_NOTICES.md. Cite this repository's tagged release using CITATION.cff, the upstream model papers, CNEMC monitoring data and ERA5-Land. Paper DOI and additional authors will be added once confirmed; no placeholder DOI is used.

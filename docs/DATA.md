# Data dictionary and provenance

The frozen V3 input consists of ten stations, 2023-01-01 through 2024-12-28, with extra weather dates for histories. All paths below are relative to data/study_lakes/v3/.

| File/column | Meaning |
|---|---|
| intermediate/readings_4h_raw_SITE.csv.gz | Selected station readings: basin, station, time, status, water temperature wt (°C), total nitrogen tn and total phosphorus tp (mg/L), algae_cells_l (cells/L) |
| qc_daily_v3_SITE.csv / y | Daily median of log10 algal density; numerical forecast target |
| wt / tn / tp | Daily water temperature and nutrient concentrations |
| *_filled | True for filled rather than genuine daily observations |
| y_obs / y_valid | Observed-day flag / valid daily-series value flag; inspect frozen QC implementation for each use |
| segment | train or eval calendar membership; does not alone determine eligibility |
| era5land_SITE.csv | Beijing-day weather aggregates: air temperature, precipitation, mean shortwave flux and derived PAR; headers retain original units |
| origins_v3.csv | Origin dates, CV/evaluation split, last-label dates and scored flags |
| summary_v3.csv / qc_removed_v3.csv | Coverage, origin counts and QC removals; readings and days are different counting units |
| manifest_v3.json | Original source hash, station coordinates, frozen input/search-space hashes and locks |

The original national archive hash identifies provenance only. The ten published raw station subsets are not claimed to have that hash. data/integrity_manifest.json identifies all distributed data files. Station-screening census summaries are provided, but exclusion decisions cannot all be recomputed without the full national source record.

The fixed-release ZIP contains 40 method–waterbody runs plus ten reference folders. test_predictions.csv contains seed-specific and final log10 predictions by origin and lead. test_scores.json retains full-precision formal metrics; the published CSV tables round to four decimals. final_fit.json contains parameter summaries, not a complete PyTorch checkpoint. run_log.json omits host machine names; scientific configuration hashes and recorded software versions remain intact.

Unresolved mean depths in the physical reference table are missing, not zero. ERA5 data are reanalysis, not a weather forecast issued at the origin. Permission provenance and limits are in DATA_LICENSE.md.

# X2 saved-array findings

Analysis completed from the immutable saved arrays and profile JSONs. No timing run, GPU operation, or X2c measurement was performed. The executable analysis is `runs/followup_v120/scripts/x2_saved_array_analysis.py`; the threshold and inclusion rules were frozen in `config/X2_ANALYSIS_PLAN.md` and committed before latency arrays were loaded.

## X2a stage means

All 16 profiles were included (8,000 profile records total). The ten instrumented stage measurements in each record are summed, and each reported share is the mean of that stage divided by the sum of the ten stage means for that profile. Rows for every PDE/method/K/cache combination are in `results/X2/X2a_stage_shares.csv`; per-profile values used for the stacked-mean plot are in `results/X2/X2a_profile_totals.csv`.

Across the eight B4 profiles, branch plus trunk averaged **4.02%** of the summed instrumented-stage budget (range **3.25–5.56%**). The P profiles averaged 3.62% (range 2.66–5.39%). In the averaged normalized stack, request preparation, field assembly/other, and encoder are the largest shares; see the per-case CSV for the exact values. These fractions are an ideal deletion bound only within the instrumented fixed sum. The profile JSONs do not contain an independently measured total profiled-request duration. Stage calls were separately synchronized and the run explicitly states that instrumentation changes execution, so the summed stages cannot be called the exact mean unprofiled request time and the deletion fraction is not a deployable p99 causal bound.

## X2b saved-array comparison

The analysis checked the recorded SHA-256 for every input array and included all rows: **120 primary rows** (20,000 requests per row, three sessions) and **360 follow-up rows** (5,000 requests per row, three sessions). `results/X2/X2b_row_statistics.csv` contains each row's median, p99, strict `latency > 2 × row median` threshold, count/share above threshold, and p99-above-threshold flag. The CSV also includes raw-file path and verified SHA-256. No row was excluded after viewing its timing values.

Using the exact matched Burgers P/B4 conditions (K=10, cache off), the frozen seed-level aggregation reproduces the cited ratio change: pool the three sessions within each seed, compute P/B4 p99, then average the three seed ratios. Primary gives **0.98591** (seed ratios 0.98405, 0.98315, 0.99054); follow-up gives **1.28925** (1.89493, 1.02801, 0.94481). Primary pooled P p99 across all three seeds and sessions is **6.7142 ms**. Follow-up P p99 by seed across its three K=10/cache-off sessions is **12.7584, 12.8189, and 13.4904 ms**. All 12 matched P/B4 checkpoints have identical SHA-256 values in the primary frozen checkpoint manifest and the follow-up timing records.

The frozen secondary-mode proxy does **not** support the expected “near 1%” description. Across all 480 rows, the median share above 2×median is **3.41%**; 93 rows fall in the prespecified descriptive 0.5–2% band. The p99 is above the threshold in 327 rows and at or below it in 153. Primary and follow-up separately have median exceedance shares of 8.83% and 2.98%. In the 18 matched Burgers P/B4 session pairs per archive, both methods' p99 are above their own threshold in every pair; therefore this matched comparison supplies no contrasting p99-location group. Its mean per-session P/B4 p99 ratio changes from 0.989 to 1.110, while the seed-pooled ratio averages above change from 0.986 to 1.289. This analysis does not identify the cause of those shifts.

For the requested stacked mean plot, the mean B4 branch-plus-trunk fraction is 4.02% across all 8 B4 profile cases, ranging from 3.25% to 5.56%. P is 3.62% across its 8 cases, ranging from 2.66% to 5.39%. The exact 16-case normalized shares are provided as tidy rows so the plot can use the measured case-level stage vectors.

The threshold is only a **secondary-mode proxy**. A threshold exceedance share and p99 position do not prove that a distinct physical mode exists, nor that it caused a p99 change. The seed-level association is descriptive and cannot establish a causal environment effect.

## Measured protocol facts

`evidence/X2/X2_ENVIRONMENT_COMPARISON.csv` records the evidence-backed comparison and identifies missing facts. The primary run used three sessions with 20,000 requests and 50 warmups per row, sampled 12,000 nominal, 4,000 coefficient-OOD, and 4,000 delay/dropout requests, and shuffled condition order per session. Follow-up used three sessions with 5,000 requests and 50 warmups per row, sampled 3,000/1,000/1,000 requests from those three roles, shuffled condition order, and explicitly varied K and cache state. Both timing paths call the shared LRU-cached nominal LQR gain implementation; follow-up measured cached and uncached operator paths. The primary metadata says `device=cuda` and `single_process_sequential=true`; follow-up records Windows WDDM with desktop applications open.

Power-plan and GPU-clock states, driver versions, and the exact runtime-library versions active at either timing event are not recorded in the inspected run evidence. The primary timing summary does not record its OS/WDDM state. They remain unknown here. Source and immutable summary hashes are enumerated in `results/X2/X2_summary.json`.

## Reproduction and validation

From `runs/followup_v120`, run:

```powershell
..\..\experiments\pdno_jevLite_20260927_v3\.venv\Scripts\python.exe scripts/x2_saved_array_analysis.py
..\..\experiments\pdno_jevLite_20260927_v3\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_x2_saved_array_analysis.py' -v
```

Both commands completed successfully. The analysis reported 16 profiles, 120 primary rows, 360 follow-up rows, and 36 paired P/B4 ratio rows; all four unit/inventory tests passed. The analysis and tests read archived arrays only.

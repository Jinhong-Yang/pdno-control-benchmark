# X2 copy-ready manuscript notes

**X2a (report only within the instrumented stage budget):** “Across the eight B4 profiles, branch plus trunk accounted for a mean 4.02% of the summed instrumented-stage time (range 3.25–5.56%). This is an ideal deletion bound within that measured stage sum; a separately recorded total profiled-request duration was unavailable, so the fraction does not establish an end-to-end latency or p99 reduction.”

Source: `runs/followup_v120/results/X2/X2a_profile_totals.csv`, rows grouped by `method=B4` (the 8 combinations of PDE, K∈{10,200}, and cache state); every profile has 500 stage records. For the stacked mean plot use the 16 condition rows and columns `*_share_of_summed_instrumented_stages` in the same CSV.

**X2b (matched Burgers K=10/cache-off):** “The primary and follow-up timing archives yield mean seed-level P/B4 p99 ratios of 0.9859 and 1.2892, respectively. These values pool three sessions within each seed before calculating each seed ratio and average the three seed ratios. Follow-up P p99 values by seed were 12.7584, 12.8189, and 13.4904 ms. The records do not establish why the ratio changed.”

Source: `runs/followup_v120/results/X2/X2_summary.json`, `burgers_matched_seed_p99`; source rows are primary arrays `session{1,2,3}_burgers_{P,B4}_s{11,23,37}.npz` and follow-up arrays of the same names with `_K10_cache0` suffix. All 12 corresponding P/B4 checkpoint SHA-256 values match between the frozen checkpoint manifest and follow-up row metadata.

**Secondary-mode proxy:** “The fraction above twice the row median was not generally near 1%: its median across 480 rows was 3.41%. The p99 exceeded this threshold in 327 rows and did not in 153; however, p99 exceeded each method’s threshold in every matched Burgers P/B4 session pair. Thus these data do not support the proposed near-1% p99-location explanation.”

Source: `runs/followup_v120/results/X2/X2_summary.json`, `threshold_statement_assessment`, and `runs/followup_v120/results/X2/X2b_row_statistics.csv`. Call the threshold a secondary-mode proxy only; these arrays cannot prove a physical mode or causal effect.

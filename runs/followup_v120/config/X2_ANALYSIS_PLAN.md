# X2 saved-array analysis plan (frozen)

Scope: X2a stage-profile aggregation and X2b comparison of the original 120-row request-latency archive with all 360 saved follow-up timing rows. This is analysis-only: do not launch benchmarks, inspect or exclude rows based on latency, or modify any old run/evidence paths.

## Frozen calculations

- X2a: process all 16 `runs/revision_v2/results/E2_profiles/session1_*.json` profiles, each expected to contain 500 records. For each record sum the ten disjoint, separately synchronized stage measurements. For P and B4 separately, report each stage's mean milliseconds and its fraction of the mean summed instrumented-stage budget. Report B4 `(branch + trunk) / summed instrumented stages` as an ideal deletion bound only within that instrumented sum. There is no separately stored total request duration for these profiled requests; do not describe this sum as observed full request latency or infer an end-to-end p99 bound.
- X2b: include every row in the primary `latency_summary_v3.json` (expected 120) and every row in follow-up `E2_TIMING.json` (expected 360); verify each referenced saved array against its recorded SHA-256 and include every row without inspection-driven exclusions. For each row compute median and p99 from the full saved `latency_ms` vector, threshold `2 * median`, count/share strictly above threshold, and boolean `p99 > threshold`. The p99 flag is the operational meaning of “p99 lies in the secondary-mode proxy”; the threshold exceedance fraction is separately reported. Do not call this proxy a proven physical/causal mode.
- For cross-era direct comparison, additionally identify matched Burgers, K=10, cache-off P/B4 rows by seed where raw arrays and metadata support it. Preserve each seed pair; provide row-level summary and paired p99 ratio. Do not infer missing setup facts.
- Environment comparison contains only claims directly supported by source scripts, immutable evidence metadata/logs, package/runtime metadata and OS/driver/library records. Missing facts are `not recorded in inspected evidence`, not guessed.
- The phrase “about one percent” is considered supported only if empirical rows have both above-threshold shares near 1% and variation in the p99-above flag (some true, some false) together with visible P/B4 p99-ratio variation by that flag. Report negative findings if these conditions fail. This is descriptive association, not a causal test.

## Integrity and outputs

Verify profile count and per-profile request count; verify source raw-file SHA-256 values before computing X2b; record input file hashes and source code hashes. Write output only under `runs/followup_v120/results/X2/` and `runs/followup_v120/evidence/X2/`. Create reproducible script under `scripts/`, meaningful tests under `tests/`, and a findings/limitations report in owned result/evidence paths. No GPU, fresh timing, X2c, or shared STATE updates.

# PDNO Control Benchmark v1.2.0 — release notes

The v1.2.0 package preserves the v1.1.0 source/evidence lineage and adds follow-up control experiments and audited saved-array analyses. DOI `10.5281/zenodo.23105442` is reserved at the archive-build stage; publication and actual-download validation are reported separately.

The package includes the X1 candidate-design analysis and audited episode arrays, X2 timing-stage and latency-distribution analyses, and X3 observer/operator error summaries with a saved CUDA gate-reproduction receipt. New raw NPZ payloads are distributed in separate GitHub evidence ZIPs; code, configurations, tests, provenance, and compact summaries are in the source archive.

The X2 analysis reports stage fractions within the sum of ten separately synchronized instrumented stages; there is no separately recorded total profiled-request duration. For the matched Burgers K=10/cache-off comparison, the mean of three seed-specific pooled-session P/B4 p99 ratios is 0.9859 in the primary archive and 1.2892 in the follow-up archive. The strict threshold `request latency > 2 × row median` is treated only as a secondary-mode proxy. The data do not support a statement that the threshold share is generally near 1%, nor do they establish actual physical modes or a causal relation to p99.

X3 reports separate observed, propagated, and total normalized errors. These quantities are not additive components of variance; their ratios are descriptive diagnostics and do not establish a causal mechanism. The GPU gate replay is included as evidence about reproduction of the saved gate values, not as a new training run or an independent test of model performance.

X1 outcomes are reported in the terminal audited CSV/JSON summaries and are conditional on the executed scenarios and controller definitions. Follow-up results do not establish deployment safety, isolated causality, or a general latency guarantee.

The v1.1.0 evidence ZIPs remain external dependencies and are identified by their original asset URLs and SHA-256 values. They are not copied into the v1.2.0 evidence ZIPs or Zenodo core archive. The four Zenodo core files are the exact v1.2.0 source ZIP, evidence manifest, checksums, and this release's scope guide.

## Intended local verification

- Verify all source and evidence archive checksums before extraction.
- Run the documented CPU implementation and saved-array checks in a fresh environment.
- Treat GPU gate replay and fresh GPU execution as distinct activities; archived CUDA replay is not a benchmark of timing or new training.
- Consult `CORE_ARCHIVE_SCOPE.md` and `REPRODUCE_FOLLOWUP.md` for retrieval and interpretation limits.

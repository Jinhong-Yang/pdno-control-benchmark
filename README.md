# PDNO Control Benchmark — version 1.2.0

Code and recorded evidence accompanying **Candidate Design, Not Predictor Accuracy, Limits Neural-Operator Control in a PDE Benchmark**, by **Hyojin Park and Jinhong Yang**. The manuscript is a submission draft, not an accepted publication.

This version preserves the v1.1.0 source and evidence lineage and adds follow-up control experiments and saved-array analyses for candidate design (X1), instrumented timing stages and latency distributions (X2), and observer/operator error diagnostics with a saved CUDA gate replay (X3). These results do not constitute a new independent test or fresh training campaign.

## Findings and scope

- X1 reports candidate-design and feedback-selection outcomes for the audited 128-scenario matched populations, plus the full Burgers baseline anchor. See `runs/followup_v120/results/X1_ANALYSIS.csv` and its JSON receipt for the exact estimates and intervals.
- X2 stage shares use the sum of ten separately synchronized measured stages as denominator. No separate total profiled-request duration was recorded. The strict `request latency > 2 × row median` statistic is a secondary-mode proxy, not proof of a physical mode. The branch/trunk share gives a fixed-instrumented-budget ideal deletion bound, not a causal deployment p99 bound.
- X3 observed, propagated, and total errors are separate diagnostics; they are not additive variance components, and their ratios do not establish causality. The saved CUDA replay checks reproduction of gate values, not new model performance.

## Source and binary evidence

The source repository is [Jinhong-Yang/pdno-control-benchmark](https://github.com/Jinhong-Yang/pdno-control-benchmark). The v1.2.0 DOI `https://doi.org/10.5281/zenodo.23105442` is reserved; its publication status must be checked at the DOI record. See `release_v1_2_0/CORE_ARCHIVE_SCOPE.md` and `EVIDENCE_MANIFEST.json` for the archive scope, exact hashes, and external dependencies.

New v1.2.0 raw NPZ payloads are distributed separately as GitHub evidence ZIPs. The eleven v1.1.0 evidence ZIPs remain external dependencies at the [v1.1.0 release](https://github.com/Jinhong-Yang/pdno-control-benchmark/releases/tag/v1.1.0); v1.2.0 does not duplicate them. Verify downloaded files and extracted members against the versioned manifests before use. The manuscript and IEEE template are maintained in a separate submission package and are not part of the software archive.

The source archive contains code, configurations, tests, compact summaries, and provenance records. It excludes raw NPZ payloads, `.git`, virtual environments, caches, logs, state files, and `WORK_ORDER.md`. The separate Zenodo core distribution consists of the source ZIP, evidence manifest, checksum list, and scope guide. A local archive build is not evidence of publication or successful download verification.

Download the current `EVIDENCE_MANIFEST.json` from the v1.2.0 release assets. The top-level file with that name inside the source tree is the retained v1.1.0 dependency manifest; it does not index new payloads. The new member-level index is `runs/followup_v120/PAYLOAD_INDEX.json`. The external v1.2.0 manifest additionally records the exact final source-archive hash and commit.

## Quick start

Use Python 3.11 in a fresh environment. On Windows, use a short root such as `D:\pdno-v1.2.0` because preserved historical paths are long.

```text
python -m venv .venv
# Activate .venv using your operating system's usual command.
python -m pip install -e ".[dev,gpu]"
python -m pytest -q
```

The historical `gpu` extra declares PyTorch; CPU-only PyTorch is sufficient for CPU checks. Consult `REPRODUCE.md` for evidence retrieval, saved-array analyses, and the optional CUDA gate replay. Recomputing archived summaries does not repeat training, closed-loop evaluation, or timing measurements.

## Citation and license

Use `CITATION.cff` for software citation metadata. Apache-2.0 applies to the original code and supplied authored assets; dependencies retain their own licenses. See `LICENSE` and `NOTICE`.

This work was supported by National IT Industry Promotion Agency(NIPA) grant funded by the Korea government(MSIT) (Development of Adaptive Physics-aware Synthetic Data Generation Technology, No.RS-2026-25621689).

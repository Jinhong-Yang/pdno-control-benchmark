# Reproduce the v3 final analysis without rerunning the experiment

The authoritative experiment root is `D:\Codex Research\Risk-Calibrated Selective Physics Correction\experiments\pdno_jevLite_20260927_v3`. The experiment is terminal and raw-frozen. **Do not execute training, calibration, locked-test, benchmark, aggregate-writing or freeze-writing scripts in that tree.** Reproduction of the present audit means reading the existing evidence and writing a new reports subdirectory, not reopening test or altering any existing artifact.

## Environment and provenance

The recorded experimental environment is local Windows, NVIDIA GeForce RTX 5080 (compute capability 12.0, 16,303 MiB reported VRAM), driver 591.86, PyTorch 2.12.0+cu130 and CUDA runtime 13.0. `evidence/preflight_v3.json` records successful forward/second-derivative CUDA smoke and the prior 47-test receipt. These were executed before this analysis. The existing isolated `.venv` is reused without changing dependencies or drivers.

The final audit actually used Python 3.11.9, NumPy 2.4.6 and a Windows platform string `Windows-10-10.0.26200-SP0`; this platform string is not a claim about the marketed Windows edition. Exact installed NumPy/SciPy/torch/pytest package versions and local task-model metadata are recorded in `RUNTIME_METADATA.json`. The final task model is **gpt-6-astra / xhigh**, verified from the local task's `turn_context`. Earlier experiment-execution alias is unknown. No new GPU work was performed by the final audit.

The workspace has no supplied Git commit for this package; file-content hashes are the provenance mechanism. Source work order: `C:\Users\WIN\Downloads\PDNO_JevLite_RTX5080_72H_실험지시서.md`. The provided reference registry is in sibling `../pdno_jevLite_20260925/references.json`; v3 does not contain its own references.json. The final analysis used the registry, v2 handoff/audit reports, v3 config/contracts/source, frozen evidence and original author placeholder. Root `AGENTS.md` applies; there is no experiment-local AGENTS.md.

## Audit actually executed

From PowerShell:

```powershell
Set-Location -LiteralPath 'D:\Codex Research\Risk-Calibrated Selective Physics Correction\experiments\pdno_jevLite_20260927_v3'
& '.venv/Scripts/python.exe' -B -X utf8 'reports/audit_frozen_v3.py' --output 'reports/analysis_20260928'
& '.venv/Scripts/python.exe' -B -X utf8 'reports/audit_sources_v3.py'
& '.venv/Scripts/python.exe' -B -X utf8 'reports/build_packet_v3.py'
& '.venv/Scripts/python.exe' -B -X utf8 'reports/validate_packet_v3.py'
```

The first command was executed once during this final task and completed with exit 0 in 54.7956559 seconds of script wall time. It produced **1,694 passing arithmetic checks**. To independently reproduce it now, replace the output with a new directory, for example `reports/reproduced_audit_01`. Existing destinations are refused. The second command is the original source/provenance authoring audit: it refreshes named report receipts and relies on the original local task metadata path, so preserve the delivered receipts and inspect that script before any reuse. It does not execute experiment modules. `build_packet_v3.py` and `validate_packet_v3.py` are report-authoring/QA helpers, not experimental runners. The builder rendered the tables and machine-readable packet once; final prose was then reviewed and edited directly. The validator checks claim support, table hashes, local links, the complete failure index and frozen inventory; it also independently checks every field in the originating task's corrected CUDA assessment against the saved raw arrays. Its final receipt is [FINAL_PACKET_VALIDATION.json](FINAL_PACKET_VALIDATION.json). An initial report-QA false positive and its repair are disclosed in the limitations file.

The arithmetic script imports no experiment package, checkpoint deserializer, model, CUDA runtime or PDE solver. It reads all 120 frozen test NPZ shards and 120 timing NPZ rows with `allow_pickle=False`; it hashes checkpoints as bytes without loading them. It reconstructs cost/tracking/violation from saved states/actions/goals, rechecks applied/projected actions, parent pairing, all registered group estimates and bootstrap intervals, 40 per-seed timing groups and deadline counts, H1's 10,000 seed/session bootstrap draws, and all 26 calibration order statistics. Its calibration calculation uses persisted parent scores; it is not an independent reconstruction of prediction tensors. The temporal-causality arrays were checked by the original registered aggregate and frozen audit; the final independent script verifies their saved count summaries rather than replaying event logic.

All outputs are under reports. Frozen evidence is never overwritten. Post-freeze heat first-step and P/B5 action-equality diagnostics are labeled descriptive and do not replace registered endpoints. The original experiment tests, training and CUDA benchmark were not rerun in this task.

## Expected outputs and exact quantitative scope

- `analysis_20260928/audit.json`: PASS; 120 test shards, 1,280 parents, 25,600 executions, 120 timing rows, 2,400,000 requests, 1,694 numerical checks, zero new model/solver/GPU executions.
- `freeze_hash_verification.csv`: 1,072 unique final-freeze entries checked. At audit entry, 1,070 matched and two disclosed administrative state/ledger entries differed; all 360 raw artifacts matched. Later authorized state updates naturally differ again and must be reported, not removed.
- `control_all_methods.csv`: 48 method/PDE/role rows; `control_per_seed.csv`: 120 rows. Parent bootstrap conditions on fitted seeds and the observed denominator.
- `latency_per_seed.csv`: 40 rows, 60,000 requests each; `latency_pooled_descriptive.csv`: 16 method/PDE rows. Pooled percentiles are computed from raw arrays, never averaged from percentiles.
- `calibration_margins.csv`: 26 rows, 64 parent maxima each, order index 61.
- `heat_constraint_diagnostic.csv`: 60 heat method/seed/role rows; all registered violations are lower-bound events already at the first advanced step; no upper-bound event.
- `P_B5_matched_actions.csv`: 18 matched role/PDE/seed rows, identifying exact Burgers action equality and four nonidentical heat parent–seed histories.
- `source_audit.json`: 24 selected operator gates fail, selected score minima reproduce, H3 comparator scores reproduce B0, all 38 pretest checkpoint hashes match. Earlier benchmark-source differences are listed.
- `cuda_diagnostic_correction.json`: the independent p99/p99.9 label audit; separate from the original supplied derived JSON and the origin-task corrected JSON.

## Immutable evidence hashes

| Artifact | SHA-256 |
| --- | --- |
| evidence/raw_evidence_freeze_v3.json | `2fc83da6f7c577d09d1c75ce44e276726a6705383e074974f94d9f321a3eea41` |
| evidence/locked_test_analysis_v3.json | `28f32812b8e1593da0a8a6f54bafc69460a0712a08e1bbc7991aa840308df3e9` |
| evidence/latency_summary_v3.json | `98555f0a22937792548b6826ea8542f3135858334fa83ebfa844d07666ced4bb` |
| evidence/latency_analysis_v3.json | `f46114814f3f0b59dc27659336c73d9b8ade9d87c0cb62b30022aad879d584f4` |
| evidence/confirmatory_checkpoint_manifest_v3.json | `c1fa3e36b0750b1c5f0a9be0132b85c3df7764b4b493938ac4ff2b364bbb1007` |
| evidence/calibration_lock_v3.json | `b8afd86d16cc1ed8cbaf31505673af942ec4446aedd44766dc8b4dd5f77e6d13` |
| config/final_spec_v3.json | `afc57e2be4d9af716c7c43ba53c3a51ccaac0a1843c8bc9363ad00b2362a8902` |
| evidence/cuda_runtime_assessment_v3.json | `f20a168cb1f8bcb9556384cf9738feba8e674f14becaa22028366ae164bb5d72` |
| evidence/cuda_runtime_assessment_v3_corrected.json | `f3b5d28f8e02b18143a8f508e9c5001ed4a93a8ef273d5a1cd979154bf214a6d` |
| AUTHOR_INPUT_REQUIRED.md | `8f9aebddd60504a5e369813e1353c494a7e90b76d00e85816d19a26408273a86` |

The raw freeze itself is unchanged. Its tree inventory includes 1,066 files; the deduplicated union with context and raw lists is 1,072. Top-level `source_audit.json` also checks the earlier 88-input pretraining and 61-input pretest manifests plus 38 checkpoint hashes. Relative to those older manifests, the appended experimental failure log and corrected host benchmark source can differ for documented reasons. Distinguish those earlier snapshots from the final raw freeze. The benchmark loader fixes changed neither completed controller trajectories nor frozen timing arrays, but they did change the benchmark source hash before successful timing collection.

## Registered historical execution commands — provenance, not instructions to rerun

The following commands identify the actual v3 script entry points recorded in state/ledger and reviewed source. Working directory is the v3 experiment root and Python is its isolated `.venv/Scripts/python.exe`. Some wrappers refuse overwrite; others, such as the registered aggregators, write named outputs, so none should be launched against this completed tree. A future independent experiment requires a separately authorized, fresh protocol/workspace; copying these commands is not that authorization.

| Stage | Historical exact entry-point command |
| --- | --- |
| Train required families | `& '.venv/Scripts/python.exe' 'scripts/run_confirmatory_models_v3.py'` |
| Freeze model selection | `& '.venv/Scripts/python.exe' 'scripts/freeze_checkpoint_manifest_v3.py'` |
| Select H3 comparator | `& '.venv/Scripts/python.exe' 'scripts/select_validation_comparator_v3.py'` |
| Generate calibration trajectories | `& '.venv/Scripts/python.exe' 'scripts/generate_calibration_data_v3.py'` |
| Generate calibration queries | `& '.venv/Scripts/python.exe' 'scripts/generate_calibration_queries_v3.py'` |
| Freeze passive forecast margins | `& '.venv/Scripts/python.exe' 'scripts/calibrate_forecast_margins_v3.py'` |
| Generate fresh locked parents | `& '.venv/Scripts/python.exe' 'scripts/generate_locked_data_v3.py'` |
| Freeze locked evaluation inputs | `& '.venv/Scripts/python.exe' 'scripts/freeze_locked_evaluation_v3.py'` |
| One-shot locked evaluation | `& '.venv/Scripts/python.exe' 'scripts/run_locked_test_once_v3.py'` |
| Registered test aggregate | `& '.venv/Scripts/python.exe' 'scripts/aggregate_locked_test_v3.py'` |
| Registered host replay | `& '.venv/Scripts/python.exe' 'scripts/benchmark_host_e2e_v3.py'` |
| Registered latency analysis | `& '.venv/Scripts/python.exe' 'scripts/analyze_latency_v3.py'` |
| Final raw freeze | `& '.venv/Scripts/python.exe' 'scripts/freeze_raw_evidence_v3.py'` |

Initial data-generation invocations and transient process command lines are not all preserved as complete shell transcripts. Generated data/teacher records and source hashes provide their exact inputs and parameters; this report does not invent missing command-line arguments. The confirmatory supervisor receipt was reconstructed after the run from persisted completion evidence and is labeled as such. The locked-test supervisor receipt and marker, completed inventories and final raw freeze supply the terminal execution trail.

## Analysis and publication boundaries

The 9,254.482-second host benchmark wall time includes campaign work and is not GPU-active occupancy. Historical total GPU-hours are unknown. Three sequential sessions and many replay requests do not create independent manufacturing environments. The root's original R1 budget and the later PDNO-specific work-order ceilings are historical constraints, not grounds to infer unused GPU time from zero GPU use in this analysis.

No full original-protocol claim is made: original parent-wise floored normalization, scheduled/contention replay, extended tail stabilization, latency-coupled control, and the conflicting closed-loop checkpoint YAML were not all executed. The current negative result is bounded to the implemented frozen benchmark. Public release, author declarations, journal format and submission are unresolved human responsibilities; authors remain AUTHOR_INPUT_REQUIRED.

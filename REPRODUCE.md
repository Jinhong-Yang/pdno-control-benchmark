# Reproducing version 1.2.0

These steps describe saved-evidence checks. They do not constitute a fresh training, control-evaluation, or latency campaign. Confirm external release availability and independently verify archive hashes before using release URLs. DOI `10.5281/zenodo.23105442` is reserved; this guide does not assert its publication status.

## 1. Retrieve and verify

Download `pdno-v1.2.0-source.zip`, `EVIDENCE_MANIFEST.json`, `SHA256SUMS.txt`, and every `pdno-v1.2.0-evidence-NN.zip` from the matching GitHub release. Also download the eleven `pdno-v1.1.0-evidence-NN.zip` assets listed as external dependencies in the manifest from the v1.1.0 GitHub release. The v1.2.0 package does not duplicate those historical ZIPs.

Verify each downloaded file against `SHA256SUMS.txt` and each evidence member against `EVIDENCE_MANIFEST.json` before extraction. Extract the source archive into a new, empty directory, then extract v1.1.0 and v1.2.0 evidence ZIPs into that source root while preserving member paths. Keep verified evidence unchanged; use a separate working copy for commands that write derived outputs.

## 2. Create an isolated environment

From the source root, use Python 3.11 and install the declared project and development dependencies:

```text
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev,gpu]"
python -m pytest -q
```

The `gpu` extra declares PyTorch and CUDA-related runtime dependencies; a CPU-built PyTorch installation is sufficient for the CPU checks below. The archived implementation tests are CPU checks. Consult the v1.1.0 `REPRODUCE.md` for historical baseline-specific checks. Do not infer exclusive-GPU timing from runs that overlapped other processes.

## 3. Recompute analyses and CPU reference propagation

After extracting the required v1.1.0 and v1.2.0 raw evidence into their manifest paths, run these commands from the source root in a disposable working copy:

```text
python runs/followup_v120/scripts/audit_x1_raw.py
python runs/followup_v120/scripts/analyze_x1.py --b0-burgers experiments/pdno_jevLite_20260927_v3/evidence/locked_test_raw_v3/locked_nominal/burgers/B0_sna.npz --b0-heat runs/revision_v2/results/E5_raw/locked_nominal/heat/B0_sna.npz
python runs/followup_v120/scripts/x2_saved_array_analysis.py
python runs/followup_v120/scripts/x3_decompose.py --evaluate
```

The X1 commands audit stored episode arrays and recompute paired summaries; they do not independently verify solver physics. X2 recomputes statistics from saved request arrays, reporting instrumented stage sums, row-median threshold shares, and p99 summaries. The strict `latency > 2 횞 row median` share is a proxy and does not prove a second physical mode. Stage shares normalize by the summed measured stages, not a separately recorded request total. `x3_decompose.py --evaluate` performs CPU reference-solver propagation for the frozen X3 cases using the archived inputs and operators; it is not only a saved-summary calculation. It does not launch CUDA, train models, or run a new control campaign.

`x2_saved_array_analysis.py` has no command-line options and directly recomputes the saved-array summaries; do not append `--help`. The X1 baseline paths above refer to the bundled v1.1.0 evidence tree after it has been extracted into the source root. Run these commands in a disposable writable copy because analysis scripts may create derived outputs.

## 4. Optional: reproduce the archived X3 CUDA gate

This optional command requires a compatible CUDA/PyTorch environment and the saved gate inputs. It reproduces the archived selected-validation values; it does not retrain weights or change inputs, endpoints, or thresholds. The command refuses to substitute CPU execution for the CUDA replay:

```text
python runs/followup_v120/scripts/x3_gate_gpu_reproduction.py --run
```

Record the device, library versions, gate receipt, and any mismatch. A CPU summary and a CUDA gate reproduction are distinct records. X3 observed, propagated, and total errors are separate diagnostics, not additive variance terms; ratios do not establish causality.

## 5. Interpretation boundaries

Keep primary-evaluation results separate from follow-up diagnostics. Preserve the original source commit and v1.1.0 assets byte-for-byte. Do not describe the timing proxy as a measured physical mode, stage-sum shares as an end-to-end latency budget, the branch/trunk fraction as a causal p99 improvement bound, or the X3 ratios as an additive error decomposition. No saved-array or gate reproduction constitutes a new independent test or establishes deployment safety.

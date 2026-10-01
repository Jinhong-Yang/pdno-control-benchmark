# PDNO Control Benchmark

Frozen implementation and summary-data companion to **Control Quality and Tail Latency of an Action-Factorized Neural Operator: A Frozen PDE Benchmark**, by **Hyojin Park and Jinhong Yang**. The manuscript is a submission draft. Software release v1.0.0 packages the completed research-v3 experiment; it does not introduce new model results.

The benchmark compares eight controller families on forced periodic Burgers and Dirichlet heat systems. It separates prediction, closed-loop utility and serialized host-ready latency. The proposed implementation did not achieve its quality/latency hypothesis: nominal cost increases over the selected modal controller were 28.04% and 13.70%; registered mean seed-specific P/B4 p99 ratios were 0.98591 and 0.99296. The original scientific disposition remains `NO_GO_V3_PROPOSED_CLAIM`.

## Archived release

Version 1.0.0: [Zenodo DOI 10.5281/zenodo.23083472](https://doi.org/10.5281/zenodo.23083472). The immutable archive corresponds to commit `b5453468157654a9b1b9c6a68619246c78afd77f`. Later citation-only documentation updates on `main` do not alter this archived version.

## Contents

| Path | Purpose |
|---|---|
| `src/pdno/` | Numerical solvers, causal observations, constraints/controllers, neural operators, training and evaluation |
| `tests/` | CPU implementation/contract tests |
| `scripts/` | Historical experiment entry points; research-v3 commands have `_v3` suffixes |
| `config/final_spec_v3.json` | Executed frozen experiment specification |
| `config/*v3.yaml` | Historical protocol configuration, including documented selection discrepancy |
| `publication/data/` | Frozen control, latency, checkpoint, calibration and training-history summaries |
| `publication/figures/` | Six main-paper figures and one supplement figure, PDF/SVG/PNG; editable Figure 1 TikZ |
| `publication/scripts/` | Portable figure/table regeneration |
| `results/` | Prior arithmetic-audit receipt, per-seed control summaries and 183-claim evidence map |
| `provenance/` | Source hashes and release boundaries |
| `reports/audit_frozen_v3.py` | Original read-only raw-array audit; requires arrays omitted from this release |

## Reproduce the figures and tables

Use a fresh Python 3.11 environment. Install pdfLaTeX with TikZ and standalone to regenerate Figure 1; all rendered assets are already supplied.

```text
python -m venv .venv
# Activate .venv using your operating system's usual command.
python -m pip install -r publication/requirements-figures.txt
python publication/scripts/build_figures.py
python publication/scripts/build_tables.py
```

Figure 1 can be edited in `publication/figures/fig01_design.tex`, then built with `python publication/scripts/build_diagram.py`. Quantitative figures read the included CSV/JSON summaries. Their scripts do not train models, solve new PDE trajectories, re-evaluate test parents or issue timing requests.

## Run implementation checks

Install PyTorch using the official instructions for your platform; a CPU build is sufficient for these tests.

```text
python -m pip install -e ".[dev,gpu]"
python -m pytest -q
```

The historical experiment used Python 3.11.9, NumPy 2.4.6 and PyTorch 2.12.0+cu130 on a Windows RTX 5080 workstation. Figure dependencies are pinned separately. CPU tests check implementation contracts; they are not a replication of GPU latency or the frozen control experiment.

## Scope of reproducibility

This release includes code and summary-level evidence. Full trajectory arrays, request-level timing arrays and model checkpoints are **not bundled**. Consequently, cloning this repository is sufficient to regenerate publication visuals and run CPU tests, but not to reproduce the entire raw-array audit or retrain the published models without additional data preparation.

The original study evaluated 1,280 parents, 25,600 controller executions and 2.4 million timing requests. Parent episodes, fitted seeds and replay requests are different units. The prior 1,694-check audit receipt is historical evidence, not a claim that this compact archive contains its complete inputs. Source locators in the 183-claim map refer to the original experiment namespace.

Historical runners are preserved for inspection. They expect staged manifests/data/evidence and some write fixed output names. Do not run them against an existing frozen experiment. Independent replication requires a fresh workspace and complete input preparation. This publication task did not rerun training, calibration, held-out control evaluation or latency benchmarking.

## Citation and license

Use `CITATION.cff` for the software authors and release version. Apache-2.0 applies to this repository's original code and assets; dependencies retain their own licenses. See `NOTICE` and `LICENSE`.

## Acknowledgment

This work was supported by National IT Industry Promotion Agency(NIPA) grant funded by the Korea government(MSIT) (Development of Adaptive Physics-aware Synthetic Data Generation Technology, No.RS-2026-25621689)

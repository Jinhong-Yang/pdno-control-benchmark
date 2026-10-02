# PDNO Control Benchmark — version 1.1.0

Code, model checkpoints and recorded evidence accompanying **When Candidate-Based Neural-Operator Control Loses to Modal Feedback: Separating Predictor Error, Controller Design, and Execution-Regime-Dependent Tail Latency in a PDE Benchmark**, by **Hyojin Park and Jinhong Yang**. The paper is a submission draft, not an accepted publication.

This version preserves the original Burgers/heat campaign and adds reference-solver candidate controllers, a candidate-count/cache timing sweep, enlarged Burgers training, a one-seed physical/ranking-loss weight sweep, and a separately generated nonnegative-initial-field heat evaluation. It includes negative and conditional findings.

## Findings and scope

- Replacing a learned forecast with reference-solver rollouts does not eliminate the modal-feedback cost advantage within the tested ten-candidate, eight-step held-action design. This does not isolate a principal causal design component.
- None of the 18 selected enlarged-budget Burgers operators reaches the 0.05 field-error gate. The added temporal targets reuse the same 512 training parents.
- On the new heat population, P nominal cost exceeds B0 by 27.46%, with a fixed-denominator paired-parent interval of 26.05–28.93%. Zero observed constraint events are not a safety guarantee.
- Connected physical-loss gradients are relatively weak in the examined diagnostics. The single-seed weight sweep changes some actions without removing the cost disadvantage.
- Timing depends on execution regime. Cached heat at 200 candidates has a mean seed-specific P/B4 p99 ratio of 0.6597, with pointwise interval 0.4728–0.9823. No interval establishes a reduction of at least 25%. These timings use the original models, not the retrained new-heat models.

## Source and binary evidence

The source repository is [Jinhong-Yang/pdno-control-benchmark](https://github.com/Jinhong-Yang/pdno-control-benchmark). The archive identifier allocated to v1.1.0 is **10.5281/zenodo.23094639**. Consult the versioned release and its manifest for availability and exact asset hashes.

The binary evidence is distributed separately from Git as **11 independent ZIP files** named `pdno-v1.1.0-evidence-01.zip` through `pdno-v1.1.0-evidence-11.zip`, plus `EVIDENCE_MANIFEST.json`. Together they contain **1,223 files and 14,281,511,705 uncompressed bytes**. Download all parts from the same version and extract them into a fresh source root. They are independent ZIP archives, not split segments of a single archive.

| Location | Contents |
|---|---|
| `src/pdno`, `tests` | Installable package and original CPU implementation tests |
| `experiments/pdno_jevLite_20260927_v3` | Preserved original sources, configurations, summaries and evidence layout |
| `runs/revision_v2` | Revision sources, configurations, training histories, numerical audits and evidence layout |
| `publication` | Fourteen rendered vector figures, editable diagram, chart inputs, tables and regeneration scripts |
| `provenance/SOURCE_FILES.json` | Staged source hashes and identified copy transformations |
| `REPRODUCE.md` | Installation, archive validation and analysis commands, with reproduction boundaries |

Binary archives supply original/revision synthetic data, raw outcomes and request timings, original protected checkpoints and all 50 selected revision checkpoints. IEEE template files, manuscript sources and author portraits are outside this code release; the submission package is separate. Preliminary E4 CPU fits are excluded from the selected revision checkpoints.

## Quick start

Use Python 3.11 in a fresh environment. On Windows, use a short root such as `D:\pdno-v1.1.0` because preserved historical paths are long.

```text
python -m venv .venv
# Activate .venv using your operating system's usual command.
python -m pip install -e ".[dev,gpu]"
python -m pip install -r publication/requirements-figures.txt
python -m pytest -q
python publication/scripts/build_tables.py
python publication/scripts/audit_revision_training_tables.py
```

The `gpu` extra is the historical name for the PyTorch dependency; a CPU PyTorch build is sufficient for implementation and checkpoint checks. Figure regeneration additionally uses a local pdfLaTeX installation with TikZ and the standalone class. Rendered assets are supplied. See `REPRODUCE.md` for raw-array and checkpoint checks after binary extraction.

## Reproduction boundaries

The original campaign contains 1,280 parents, 25,600 controller executions and 2.4 million timing requests. Revision outcome arrays contain 21,760 executions, and the additional timing sweep contains 1.8 million requests. Executions, training seeds and timing requests are not additional independent parents.

Author-workspace audits reconstructed recorded arithmetic and intervals. Replaying these audits does not rerun model training or closed-loop dynamics and does not establish control benefit. Historical runners retain fixed output names and workspace assumptions; they are not a one-command fresh GPU replication interface. Do not run the historical supervisor over the extracted evidence. Use a separate output namespace for a new campaign.

The original release [v1.0.0, DOI 10.5281/zenodo.23083472](https://doi.org/10.5281/zenodo.23083472) remains an immutable code/summary archive without binary evidence. Its source commit is `b5453468157654a9b1b9c6a68619246c78afd77f`; v1.1.0 does not replace that record.

## Citation, license and support

Use `CITATION.cff` for authors and version. Apache-2.0 applies to the original code and supplied authored assets; dependencies retain their own licenses. See `LICENSE` and `NOTICE`.

This work was supported by National IT Industry Promotion Agency(NIPA) grant funded by the Korea government(MSIT) (Development of Adaptive Physics-aware Synthetic Data Generation Technology, No.RS-2026-25621689)

# Reproducing version 1.1.0

These instructions distinguish summary regeneration, recorded-evidence verification, checkpoint loading and fresh experimentation. Use a new directory rather than an existing research workspace. The supplied manuscript is a draft.

## 1. Source and environment

Obtain the v1.1.0 source. Use Python 3.11 and a short root on Windows, for example `D:\pdno-v1.1.0`. Create and activate a separate environment, then run from that source root:

```text
python -m venv .venv
# Activate .venv using your operating system's usual command.
python -m pip install -e ".[dev,gpu]"
python -m pip install -r publication/requirements-figures.txt
python -m pytest -q
```

The historical `gpu` extra installs PyTorch. CPU-only PyTorch is sufficient for the checks below. The training/timing campaign used Windows, Python 3.11.9 and PyTorch 2.12.0+cu130 on an RTX 5080; CPU checks do not replicate that GPU timing environment. Figure dependencies are pinned in `publication/requirements-figures.txt`.

## 2. Validate and extract the evidence

Download `EVIDENCE_MANIFEST.json` and every evidence ZIP in its `archives` list from the same v1.1.0 release. There are 11 independent ZIP files containing 1,223 payload files, totaling 14,281,511,705 uncompressed bytes. The manifest's build status describes packaging provenance and does not itself certify external publication or a scientific outcome.

Replace `ARCHIVE_DIRECTORY` below with the download directory:

```text
python runs/revision_v2/release_support/verify_evidence.py ARCHIVE_DIRECTORY/EVIDENCE_MANIFEST.json --archives ARCHIVE_DIRECTORY
```

Extract every ZIP into the source-tree root while preserving its member paths. Then verify extracted sizes and hashes:

```text
python runs/revision_v2/release_support/verify_evidence.py ARCHIVE_DIRECTORY/EVIDENCE_MANIFEST.json --root .
```

Keep this verified copy intact. Run scripts that write derived outputs in a separate disposable copy. Do not extract evidence over an existing experiment.

## 3. Regenerate figures and tables

From the source root:

```text
python publication/scripts/build_figures.py
python publication/scripts/build_tables.py
python publication/scripts/audit_completed_tables.py
python publication/scripts/audit_revision_timing_tables.py
python publication/scripts/audit_revision_training_tables.py
```

Figure 1 uses pdfLaTeX with TikZ and the standalone class. Its editable source is `publication/figures/fig01_design.tex`; PDF/SVG/PNG renders are supplied. The other figures use supplied CSV/JSON summaries. These commands do not train models, reopen evaluation decisions or benchmark a GPU.

The completed-table auditor covers 22 supplementary tables and the main nominal table, including the two oracle rows. Separate auditors cover 14 timing tables and 35 revision training/control tables. Their checks establish keyed transcription, aggregation and rounding against source summaries, not the scientific validity of a claim or the PDF layout. The code release intentionally excludes manuscript sources and template assets, so evidence-number generation skips manuscript scanning when both main and supplement are absent.

## 4. Load the archived checkpoints on CPU

After evidence extraction:

```text
python runs/revision_v2/tests/check_checkpoint_relocation.py
python runs/revision_v2/tests/check_revision_checkpoint_relocation.py
```

The first command checks twelve original seed-11 controller/PDE cases. The second requires all 50 selected revision checkpoints: 21 E3, ten E4-CUDA and 19 E5, including four observers. Both copy code and weights to a temporary directory, disable GPU visibility, block checkpoint reads outside that copy and evaluate one training-observation fixture per model. This verifies loading and finite output shapes. It does not evaluate closed-loop quality, optimizer continuation or GPU performance. The tests write receipts below `runs/revision_v2/results`.

## 5. Recompute analyses from recorded arrays

Use a disposable copy because these scripts write derived files under their historical names. The following commands process saved observations/results rather than generating new trajectories or fitting models:

```text
python runs/revision_v2/scripts/audit_revision_raw.py
python runs/revision_v2/scripts/audit_revision_intervals.py
python runs/revision_v2/scripts/audit_e5_policy_freeze.py
```

The raw audit reconstructs state/action costs, tracking, strict/tolerance violations, amounts, timing quantiles and counts, and preservation hashes. The completed author-workspace result was 3,812 checks over 102 outcome arrays and 360 timing arrays. The interval audit independently checks all 72 revision summary rows and 360 estimate/interval values with 10,000 paired-parent draws. Parent seeds and repeated methods are not treated as independent environments. Fixed-denominator and jointly resampled-denominator intervals are pointwise and conditional on the available trained models.

The E5 audit checks 19 checkpoint hashes, 35 source/configuration hashes, split identities and linkage between the local seal and subsequent test generation. It does not certify an external timestamp or preregistration. The original campaign's independent arithmetic audit is preserved under `experiments/pdno_jevLite_20260927_v3/reports`; its recorded 1,694 checks are separate from the revision audit.

Additional derived-export entry points are:

```text
python runs/revision_v2/scripts/export_training_evidence.py
python runs/revision_v2/scripts/summarize_e2.py
python runs/revision_v2/scripts/summarize_revision.py
```

The training exporter needs checkpoint companion `.summary.json` files from both campaign snapshots. Exported values may be compared to the supplied CSVs; receipts contain local paths and times and need not be byte-identical after relocation. Run exports in a separate copy, since overwriting an input receipt can invalidate a prior receipt's file hash even when scientific values agree. Retain the immutable provided manifests for comparison.

## 6. What requires a new experiment

The historical runners expect staged data, completion receipts, fixed output locations or author-workspace dependencies. They document the executed campaign and are not a one-command fresh replication interface. Do not launch the historical supervisor against the supplied evidence tree. A new training/control/timing campaign needs a distinct output namespace, validation-only selection, policy freezing and sequential GPU scheduling.

No E3 operator passed the field-error gate, so the absence of E3 closed-loop arrays is intentional. E4 is a seed-11 sweep on original parents; E5 uses new heat parents and retrained models. E2 timing uses original P/B4 weights. Do not join the E2 latency and E5 control numbers as measurements of one deployed configuration.

Timing sessions shared a process with persistent gain caches, ran on Windows with unfixed clocks and retained outliers. Instrumented stage profiles add synchronization; marginal stage quantiles cannot be summed into end-to-end p99. Neither CPU tests nor archived-array checks reproduce real deployment deadlines or a safety guarantee.

The archive identifier allocated to this version is `10.5281/zenodo.23094639`. The original v1.0.0 record and DOI `10.5281/zenodo.23083472` remain unchanged. Verify actual release availability and downloaded bytes independently of the author-workspace receipts retained in this source tree.

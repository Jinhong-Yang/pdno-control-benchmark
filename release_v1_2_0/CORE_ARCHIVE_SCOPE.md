# PDNO Control Benchmark v1.2.0: core archive scope

DOI `10.5281/zenodo.23105442` is reserved. This guide specifies the core archive contents and retrieval boundaries; it is not evidence of an upload, public release, or DOI activation.

The Zenodo core distribution contains exactly these four files:

- `pdno-v1.2.0-source.zip`: a ZIP created from the exact, clean v1.2.0 source Git commit after the source stage has been reviewed and committed;
- `EVIDENCE_MANIFEST.json`: source-commit identity, the new raw-payload archive/member hashes, and the exact external v1.1.0 evidence dependencies;
- `SHA256SUMS.txt`: checksums for the source ZIP, new evidence ZIPs, manifest, and scope guide;
- this scope and retrieval guide.

The source tree includes the v1.2.0 follow-up scripts, source modules, configurations, tests, saved-array summaries, and provenance receipts under `runs/followup_v120/`. It excludes `.git`, virtual environments, caches, logs, state files, `WORK_ORDER.md` (editorial task instructions), and raw NPZ payloads. The source ZIP is built from the exact reviewed source commit, not from a working tree.

Release staging reuses the existing clean v1.1.0 checkout at `release_staging/github_v110`, whose `main` branch and peeled `v1.1.0` tag must both resolve to `bc6de067d3be547ed01345b48625eb52bcdc0666`. After that base guard passes, v1.2.0 files are staged on a new `release-v1.2.0` branch. The existing `v1.1.0` tag is immutable and must remain at its recorded commit. The source checkout's `.gitattributes` `* -text` rule is preserved to retain byte-level hashes. The source archive is built only after the new branch has been reviewed, committed, and is clean.

New raw NPZ result payloads are distributed as separate `pdno-v1.2.0-evidence-NN.zip` GitHub release assets. The v1.2.0 manifest records each member's repository-relative path, uncompressed byte size, SHA-256, and archive assignment. These raw arrays are not included in the Zenodo core archive. The optional `pdno-v1.2.0-zenodo-core.zip` is a local convenience copy of the four core files and is not required for the canonical four-file distribution.

The eleven `pdno-v1.1.0-evidence-NN.zip` assets remain exact external dependencies at the v1.1.0 GitHub release. Their URLs, sizes, and SHA-256 values are copied from the immutable v1.1.0 evidence manifest. They are not downloaded, recompressed, or duplicated in v1.2.0. Retrieve them from:

https://github.com/Jinhong-Yang/pdno-control-benchmark/releases/tag/v1.1.0

Verify each downloaded asset against the v1.2.0 manifest before extraction. The archived v1.1.0 source commit remains `bc6de067d3be547ed01345b48625eb52bcdc0666`; its `v1.1.0` Git tag and existing GitHub/Zenodo records must remain unchanged.

The core DOI record identifies the source and reproducibility metadata. It does not itself preserve new or historical raw arrays; those remain on GitHub with exact external checksums. A manifest entry for a GitHub asset does not imply that the asset is contained in the Zenodo archive.

## Exclusions and limits

- No manuscript, IEEE template, author portrait, or editorial task instruction is part of the software archive.
- No private host logs, credentials, virtual environments, or unrelated project files are included.
- No DOI or release is described as public until the corresponding external record and downloadable assets are independently verified.
- X1/X2/X3 results are conditional on their saved arrays, sampled scenarios, measured environment, and stated comparison design. The package does not claim deployment safety, a general p99 bound, an actual physical latency mode, or causal variance decomposition.

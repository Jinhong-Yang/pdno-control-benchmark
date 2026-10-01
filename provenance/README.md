# Source lineage

SOURCE_MANIFEST.json hashes each released file copied from the frozen v3 implementation or the publication package. `frozen-v3/` refers to the completed 2026-09-27 experiment namespace. `publication/` refers to the 2026-10-01 manuscript asset package. These origin labels are portable identifiers, not host filesystem paths.

The frozen raw-evidence manifest has SHA-256 `2fc83da6f7c577d09d1c75ce44e276726a6705383e074974f94d9f321a3eea41`. The original independent audit reported 1,694 arithmetic checks. Its summary is preserved in results/audit.json; this release does not rerun or claim to include the raw arrays needed for that audit. Local administrative state/ledger changes were distinct from the frozen scientific files.

The original checkpoint YAML and final training-summary selection differ. The former is retained as historical configuration; final_spec_v3.json and the manuscript describe the executed protocol. Neither source publication nor passing CPU tests resolves the experiment's optimization-sufficiency limitation.

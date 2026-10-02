from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "results/X1/burgers_nominal/K10_H8/n384.npz"
FULL_RECORD = FULL.with_suffix(".record.json")
OUT = ROOT / "results/X1/burgers_nominal/K10_H8/n128.npz"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    check = ROOT / "results/X1_BASELINE_VERIFICATION.json"
    if not check.exists() or json.loads(check.read_text(encoding="utf-8")).get("status") != "PASS":
        raise RuntimeError("Do not derive a paired subset until the full384 anchor passes")
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT}")
    with np.load(FULL, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    if len(a["parent_id"]) != 384:
        raise ValueError("full384 baseline anchor required")
    arrays = {k: v[:128] if v.ndim and v.shape[0] == 384 else v for k, v in a.items()}
    np.savez_compressed(OUT, **arrays)
    prior = json.loads(FULL_RECORD.read_text(encoding="utf-8"))
    ids = arrays["parent_id"].astype(str)
    record = dict(prior)
    record.update({
        "status": "DERIVED_MATCHED_SUBSET",
        "parent_count": 128,
        "scenario_id_order": "first128 from full384 anchor sorted lexical by actual scenario ID",
        "derived_from": str(FULL),
        "derived_from_sha256": sha(FULL),
        "baseline_verification_sha256": sha(check),
        "scenario_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "result_sha256": sha(OUT),
        "note": "No new PDE simulation; deterministic matched cohort extracted after the predeclared full384 anchor passed.",
    })
    OUT.with_suffix(".record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "n": len(ids), "path": str(OUT), "sha256": record["result_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

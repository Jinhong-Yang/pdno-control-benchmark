"""Freeze all 36 selected learned checkpoints and the two shared observers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("P", "P-no-rank", "B4", "B5", "B2", "B3")
PDEs = ("burgers", "heat")
SEEDS = (11, 23, 37)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    out = ROOT / "evidence" / "confirmatory_checkpoint_manifest_v2.json"
    if out.exists():
        raise SystemExit("v2 checkpoint manifest already exists; refusing to overwrite")
    progress_path = ROOT / "evidence" / "v2_confirmatory_training_progress.json"
    selection_path = ROOT / "evidence" / "validation_checkpoint_selection_v2.json"
    if not progress_path.is_file() or not selection_path.is_file():
        raise SystemExit("confirmatory training completion or checkpoint selection evidence missing")
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if progress.get("status") != "TRAINING_COMPLETE":
        raise SystemExit("mandatory training is not marked complete")
    if selection.get("test_opened") is not False:
        raise SystemExit("selection evidence does not prove test remains sealed")
    rows = []
    expected = {(pde, method, seed) for pde in PDEs for method in METHODS for seed in SEEDS}
    selected = {(r["pde"], r["method"], int(r["seed"])): r for r in selection.get("selected", [])}
    if selected.keys() != expected:
        raise SystemExit(f"selection coverage mismatch: missing={expected-selected.keys()}, extra={selected.keys()-expected}")
    for pde, method, seed in sorted(expected):
        path = ROOT / "runs" / "confirmatory_v2" / pde / f"{method}_s{seed}.pt"
        if not path.is_file() or sha(path) != selected[(pde, method, seed)]["official_checkpoint_sha256"]:
            raise SystemExit(f"selected official checkpoint missing/hash mismatch: {path}")
        item = torch.load(path, map_location="cpu", weights_only=False)
        meta = item.get("metadata", {})
        if (meta.get("pde"), meta.get("method"), int(meta.get("seed", -1))) != (pde, method, seed):
            raise SystemExit(f"checkpoint metadata identity mismatch: {path}: {meta}")
        rows.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                     "sha256": sha(path), "method": method, "pde": pde, "seed": seed,
                     "selection_status": selected[(pde, method, seed)]["selection_status"],
                     "selected_update": selected[(pde, method, seed)]["update"],
                     "metadata": meta})
    observers = []
    for pde in PDEs:
        path = ROOT / "runs" / "confirmatory_v2_training" / "observer" / f"{pde}_seed7.pt"
        if not path.is_file():
            raise SystemExit(f"shared observer missing: {path}")
        item = torch.load(path, map_location="cpu", weights_only=False)
        if item.get("metadata", {}).get("method") != "observer" or item["metadata"].get("pde") != pde:
            raise SystemExit(f"shared observer metadata mismatch: {path}")
        observers.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
                          "sha256": sha(path), "method": "observer", "pde": pde,
                          "seed": 7, "metadata": item["metadata"]})
    if len(rows) != 36 or len(observers) != 2:
        raise SystemExit("expected 36 learned checkpoints and 2 observers")
    manifest = {"protocol": "PDNO_JevLite_RTX5080_72H_frozen_spec_v2",
        "status": "FROZEN_CONFIRMATORY_CHECKPOINTS", "test_opened": False,
        "learned_checkpoint_count": 36, "observer_count": 2,
        "training_progress_sha256": sha(progress_path),
        "selection_evidence_sha256": sha(selection_path),
        "checkpoint_selection_status_counts": {
            status: sum(row["selection_status"] == status for row in rows)
            for status in sorted({row["selection_status"] for row in rows})},
        "checkpoints": rows, "observers": observers}
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "learned_checkpoints": len(rows),
                      "observers": len(observers), "selection_status_counts": manifest["checkpoint_selection_status_counts"],
                      "manifest": str(out), "sha256": sha(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

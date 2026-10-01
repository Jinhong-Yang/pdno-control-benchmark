"""Hash the fail-inclusive v2 experiment contract before confirmatory training."""
from __future__ import annotations

import hashlib
import json
import argparse
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amend", action="store_true", help="write a superseding hash lock without editing the first freeze")
    args = parser.parse_args()
    original_target = ROOT / "evidence" / "pretraining_spec_freeze_v2.json"
    target = ROOT / "evidence" / ("pretraining_spec_freeze_v2_amended.json" if args.amend else "pretraining_spec_freeze_v2.json")
    if target.exists():
        raise SystemExit(f"pretraining spec freeze already exists: {target}")
    if args.amend and not original_target.is_file():
        raise SystemExit("cannot amend before an original freeze exists")
    locked_root = ROOT / "data" / "locked_v2"
    if locked_root.exists() and any(locked_root.rglob("*.npz")):
        raise SystemExit("locked v2 data exists before pretraining spec freeze")
    required = [
        ROOT / "config" / "final_spec_v2.json",
        ROOT / "config" / "evaluation_addendum_v2.yaml",
        ROOT / "config" / "checkpoint_selection_v2.yaml",
        ROOT / "config" / "research_version_2_preregistration.yaml",
        ROOT / "config" / "frozen_experiment.yaml",
        ROOT / "contracts" / "input_schema.json",
        ROOT / "scripts" / "run_confirmatory_models_v2.py",
        ROOT / "scripts" / "select_validation_checkpoints_v2.py",
        ROOT / "scripts" / "select_validation_comparator_v2.py",
        ROOT / "src" / "pdno" / "training" / "confirmatory.py",
        ROOT / "src" / "pdno" / "training" / "fit.py",
        ROOT / "src" / "pdno" / "evaluation" / "closed_loop.py",
        ROOT / "evidence" / "v2_evaluation_budget_reconciliation.json",
        ROOT / "evidence" / "v2_rev1_pilot_g2_validation_metrics.json",
        ROOT / "AUTHOR_INPUT_REQUIRED.md",
    ]
    required.extend(sorted((ROOT / "src" / "pdno").rglob("*.py")))
    required.extend(sorted(path for path in (ROOT / "data" / "train").rglob("*") if path.is_file()))
    required.extend(sorted(path for path in (ROOT / "data" / "validation").rglob("*") if path.is_file()))
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing freeze inputs: " + ", ".join(str(path) for path in missing))
    unique = sorted(set(required))
    previous = json.loads(original_target.read_text(encoding="utf-8")) if args.amend else None
    current_inputs = {str(path.relative_to(ROOT)): sha(path) for path in unique}
    changed = []
    if previous is not None:
        old_inputs = previous.get("inputs_sha256", {})
        changed = [{"path": name, "previous_sha256": old_inputs.get(name), "current_sha256": digest}
                   for name, digest in current_inputs.items() if old_inputs.get(name) != digest]
    manifest = {"status": "PRETRAINING_SPEC_FROZEN_TEST_SEALED",
        "frozen_at_local": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "spec_path": str((ROOT / "config" / "final_spec_v2.json").relative_to(ROOT)),
        "spec_sha256": sha(ROOT / "config" / "final_spec_v2.json"),
        "inputs_sha256": current_inputs,
        "train_validation_data_manifest_sha256": {
            str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / "data").rglob("*.record.json"))
            if "train" in path.parts or "validation" in path.parts},
        "calibration_generated": False, "locked_test_generated": False, "locked_test_opened": False,
        "g2_pilot_disposition": "G2_FAIL_RETAINED_NO_THRESHOLD_CHANGE",
        "author_input_required": True,
        "supersedes_freeze_sha256": sha(original_target) if previous is not None else None,
        "changed_freeze_inputs": changed,
        "amendment_reason": ("selector finite-output implementation and train/validation data payload hashes added before any confirmatory fit"
                             if previous is not None else None)}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(target)
    print(json.dumps({"status": manifest["status"], "spec_sha256": manifest["spec_sha256"],
                      "frozen_input_count": len(unique), "output": str(target)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify required confirmatory checkpoint coverage and freeze per-file hashes."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
EXPECTED={(m,p,s) for m in ("P","B4","B5","B2","B3") for p in ("burgers","heat") for s in (11,23,37)}

def sha(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def main():
    root=ROOT/"runs"/"confirmatory_frozen_v1"
    rows=[]; seen=set()
    for path in sorted(root.rglob("*.pt")):
        item=torch.load(path,map_location="cpu",weights_only=False)
        meta=item["metadata"]
        method,pde=meta["method"],meta["pde"]
        if method=="observer":
            key=(method,pde,7)
        else:
            key=(method,pde,int(meta["seed"]))
            if key not in EXPECTED: raise SystemExit(f"unexpected checkpoint identity {key} at {path}")
            seen.add(key)
        summary_path=path.with_suffix(".summary.json")
        rows.append({"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"sha256":sha(path),
                     "method":method,"pde":pde,"seed":int(meta.get("seed",7)),
                     "summary":json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None})
    missing=sorted(EXPECTED-seen)
    if missing: raise SystemExit(f"missing required confirmatory checkpoints: {missing}")
    if len(rows)!=32: raise SystemExit(f"expected 32 checkpoints (30 learned + 2 observer), found {len(rows)}")
    progress=ROOT/"evidence"/"confirmatory_training_progress.json"
    p=json.loads(progress.read_text(encoding="utf-8"))
    if p.get("status")!="TRAINING_COMPLETE": raise SystemExit("training progress is not marked complete")
    manifest={"protocol":"PDNO_JevLite_RTX5080_72H_frozen_v1","status":"CHECKPOINTS_FROZEN",
              "test_opened":False,"learned_model_count":30,"observer_count":2,"checkpoints":rows,
              "training_progress_sha256":sha(progress),"missing_expected":missing}
    out=ROOT/"evidence"/"confirmatory_checkpoint_manifest.json"
    out.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"checkpoint_count":len(rows),"learned_model_count":sum(r["method"]!="observer" for r in rows),
                      "observers":sum(r["method"]=="observer" for r in rows),"missing":missing,
                      "manifest":str(out),"training_progress_sha256":manifest["training_progress_sha256"]},indent=2))

if __name__=="__main__": main()

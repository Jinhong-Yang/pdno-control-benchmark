"""CPU relocation check of selected revision weights; default requires all 50.

--allow-partial is an authoring support option and cannot certify a release.
Only completed .summary.json companions identify immutable selected weights.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np

R = Path(__file__).resolve().parents[1]
ROOT = R.parents[1]
OLD = ROOT / "experiments/pdno_jevLite_20260927_v3"
KEYS = ["sensor_value", "sensor_mask", "sensor_age", "instrument_image",
        "image_mask", "goal_coefficients", "material_context", "previous_applied_action",
        "applied_action_history", "image_age", "image_valid", "candidate_action"]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
CHILD = r'''
from pathlib import Path
import hashlib,json,sys
import numpy as np
import torch
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/"src"))
torch.set_num_threads(1);torch.set_num_interop_threads(1)
assert not torch.cuda.is_available(), "CPU-only support environment required"
from pdno.evaluation.closed_loop import load_controller
from pdno.training.confirmatory import _Observer
from pdno.models.encoders import load_state_with_normalizer_defaults
from pdno.training.fit import _grid
original_load=torch.load
reads=[]
def guarded_load(path,*args,**kwargs):
    p=Path(path).resolve()
    assert p.is_relative_to((root/"models").resolve()),"External checkpoint read blocked"
    reads.append(p.name)
    return original_load(path,*args,**kwargs)
torch.load=guarded_load
rows=[]
for item in json.loads((root/"inventory.json").read_text()):
    pde,method,seed=item["pde"],item["method"],item["seed"]
    with np.load(root/item["fixture"],allow_pickle=False) as z:
        obs={k:torch.as_tensor(z[k]) for k in z.files}
    actions=obs.pop("candidate_action").float()
    x,tau=_grid(pde,torch.device("cpu"))
    checkpoint=root/"models"/item["weight"]
    if method=="observer":
        model=_Observer(pde)
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        load_state_with_normalizer_defaults(model,saved["state_dict"])
        model.eval()
        with torch.no_grad(): output=model(obs,x)
        expected=(1,128)
    else:
        model=load_controller(method,pde,seed,checkpoint,torch.device("cpu"))
        with torch.no_grad(): output=model(obs) if method in ("B2","B3") else model(obs,actions,x,tau)
        expected=(1,2) if method in ("B2","B3") else (1,10,8,128)
    assert tuple(output.shape)==expected and torch.isfinite(output).all(),item
    rows.append({**item,"shape":list(output.shape),"output_sha256":hashlib.sha256(output.numpy().tobytes()).hexdigest()})
assert len(reads)==len(rows)==len(set(reads))
print(json.dumps({"rows":rows,"checkpoint_reads":reads,"gpu_operations":0}))
'''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    selected = []
    counts = {}
    for stage, folder, required in (("E3", "E3", 21), ("E4", "E4_cuda", 10), ("E5", "E5", 19)):
        summaries = sorted((R/"checkpoints"/folder).rglob("*.summary.json"))
        counts[stage] = {"completed": len(summaries), "required": required}
        assert len(summaries) <= required, (stage, len(summaries))
        if not args.allow_partial:
            assert len(summaries) == required, ("Incomplete revision checkpoint set", stage, counts[stage])
        for summary_path in summaries:
            summary = json.loads(summary_path.read_text())
            checkpoint = summary_path.with_name(summary_path.name.removesuffix(".summary.json")+".pt")
            assert checkpoint.is_file()
            selected.append((stage, summary_path, checkpoint, summary))
    assert selected, "No completed weights"
    base = (ROOT/"tmp/release_checks").resolve()
    base.mkdir(parents=True, exist_ok=True)
    assert base.is_relative_to(ROOT.resolve())
    hashes, items = {}, []
    with tempfile.TemporaryDirectory(prefix="rev-", dir=base) as td:
        target = Path(td).resolve()
        assert target.is_relative_to(base)
        shutil.copytree(R/"src", target/"src", ignore=shutil.ignore_patterns("__pycache__"))
        for path in sorted((R/"src").rglob("*.py")):
            hashes[str(path.relative_to(ROOT))] = sha(path)
            assert sha(target/"src"/path.relative_to(R/"src")) == hashes[str(path.relative_to(ROOT))]
        (target/"models").mkdir()
        fixtures = {}
        for index, (stage, summary_path, checkpoint, summary) in enumerate(selected):
            pde = summary["pde"]
            group = ("new_heat" if stage=="E5" else "original") + "_" + pde
            if group not in fixtures:
                source = (R/"data/positive_heat_queries/train/heat/teacher_queries.npz") if stage=="E5" else OLD/f"data/queries_v3/train/{pde}/teacher_queries.npz"
                fixtures[group] = group+".npz"
                with np.load(source, allow_pickle=False) as z:
                    np.savez(target/fixtures[group], **{k:z[k][:1] for k in KEYS})
                hashes[str(source.relative_to(ROOT))] = sha(source)
                hashes["relocated_fixture/"+fixtures[group]] = sha(target/fixtures[group])
            name = f"{index:02d}_{stage}_{checkpoint.name}"
            before = sha(checkpoint)
            shutil.copy2(checkpoint, target/"models"/name)
            assert sha(target/"models"/name) == sha(checkpoint) == before
            hashes[str(checkpoint.relative_to(ROOT))] = before
            hashes[str(summary_path.relative_to(ROOT))] = sha(summary_path)
            items.append({"stage":stage, "pde":pde, "method":summary["method"], "seed":summary["seed"],
                          "selected_update":summary["best_step"], "weight":name,
                          "source":str(checkpoint.relative_to(ROOT)), "fixture":fixtures[group]})
        (target/"inventory.json").write_text(json.dumps(items))
        (target/"check.py").write_text(CHILD)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONPATH="")
        process = subprocess.run([sys.executable,"-B","-X","utf8",str(target/"check.py")],
                                 cwd=target,env=env,capture_output=True,text=True,encoding="utf-8",timeout=180)
        if process.returncode:
            raise RuntimeError(process.stdout+"\n"+process.stderr)
        result = json.loads(process.stdout)
    complete = all(v["completed"]==v["required"] for v in counts.values())
    receipt = {"status":"PASS_ALL_50_REVISION_CHECKPOINTS_CPU_RELOCATION" if complete else "PASS_PARTIAL_REVISION_CPU_RELOCATION",
               "complete_checkpoint_inventory":complete, "counts":counts, "external_checkpoint_reads_allowed":False,
               "input_sha256":hashes, "test_sha256":sha(Path(__file__)), **result,
               "limits":"One training-observation forward pass per selected model using relocated code and weights. No closed-loop evaluation, GPU performance, full fresh-install or downloaded-archive certification."}
    output = R/"results"/("REVISION_CHECKPOINT_RELOCATION.json" if complete else "REVISION_CHECKPOINT_RELOCATION_PARTIAL.json")
    output.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps({"status":receipt["status"],"models":len(result["rows"]),"counts":counts}))

if __name__ == "__main__":
    main()

"""Train/validation-only interpolation diagnostics for the multimodal initial field."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _nrmse(pred, truth):
    return float(np.sqrt(np.square(pred - truth).sum() / max(np.square(truth).sum(), 1e-12)))


def audit(path: Path, pde: str):
    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    truth = d["initial_field"].astype(np.float64)
    n = truth.shape[-1]
    if pde == "burgers":
        x = np.arange(n) / n
        image_x = np.arange(64) / 64
        sensor_x = np.arange(16) / 16
        scale = 1.5
    else:
        x = np.arange(1, n + 1) / (n + 1)
        image_x = np.arange(1, 65) / 65
        sensor_x = (np.linspace(0, 255, 16) + 1) / 257
        scale = 1.0
    image_pred, sensor_pred, combined_pred = [], [], []
    for i in range(len(truth)):
        valid_cols = d["image_mask"][i, 0].mean(axis=0) >= 0.5
        if d["image_valid"][i] and valid_cols.any():
            img = d["instrument_image"][i, 0].astype(np.float64)
            y = np.median(img[:, valid_cols], axis=0)
            xi = image_x[valid_cols]
            vals = (y - 0.5) * (2 * scale)
            if pde == "burgers":
                pi = np.interp(x, xi, vals, period=1.0)
            else:
                pi = np.interp(x, xi, vals, left=0.0, right=0.0)
        else:
            pi = np.zeros(n)
        valid = d["sensor_mask"][i, -1].astype(bool)
        sx, sy = sensor_x[valid], d["sensor_value"][i, -1, valid].astype(np.float64)
        if len(sx) >= 2:
            ps = np.interp(x, sx, sy, period=1.0) if pde == "burgers" else np.interp(x, sx, sy, left=0.0, right=0.0)
        else:
            ps = np.zeros(n)
        if valid_cols.any() and d["image_valid"][i]:
            pc = pi
        else:
            pc = ps
        image_pred.append(pi); sensor_pred.append(ps); combined_pred.append(pc)
    image_pred, sensor_pred, combined_pred = map(np.asarray, (image_pred, sensor_pred, combined_pred))
    return {"pde":pde,"split":path.parent.parent.name,"query_count":len(truth),
            "parent_count":len(set(d["parent_id"].astype(str).tolist())),
            "image_interpolation_nrmse":_nrmse(image_pred,truth),
            "sensor_interpolation_nrmse":_nrmse(sensor_pred,truth),
            "image_with_sensor_fallback_nrmse":_nrmse(combined_pred,truth)}


def main():
    rows=[]
    for pde in ("burgers","heat"):
        for split in ("train","validation"):
            rows.append(audit(ROOT/"data"/"queries_v2"/split/pde/"teacher_queries.npz",pde))
    out=ROOT/"evidence"/"observation_interpolation_floor_v2.json"
    out.write_text(json.dumps(rows,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()

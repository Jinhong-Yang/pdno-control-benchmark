"""Run the fixed 50-warm-up/200-measured pilot for all learned method classes."""
from __future__ import annotations

import json
from pathlib import Path
import ctypes
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from pdno.training.fit import train_bc, train_b3, train_operator


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def _peak_rss_mib() -> float:
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_ulong]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    process = kernel32.GetCurrentProcess()
    if not psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb):
        return float("nan")
    return counters.PeakWorkingSetSize / 2**20


def main() -> int:
    out_root = ROOT / "runs" / "runtime_pilot_50_200_v2"
    rows = []
    for pde in ("burgers", "heat"):
        query = ROOT / "data" / "queries_v2" / "train" / pde / "teacher_queries.npz"
        batch_size = 64
        by_method = {}
        for method in ("P", "B4", "B5", "B2", "B3"):
            checkpoint = out_root / f"{method}_{pde}_s11.pt"
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            if method in {"P", "B4", "B5"}:
                summary = train_operator(query, pde, method, 11, 250, batch_size, checkpoint,
                                         "cuda", timing_warmup=50)
            elif method == "B2":
                summary = train_bc(query, pde, 11, 250, batch_size, checkpoint, "cuda", timing_warmup=50)
            else:
                summary = train_b3(query, pde, 11, 250, checkpoint,
                                   by_method["B2"]["checkpoint"], by_method["B4"]["checkpoint"],
                                   "cuda", timing_warmup=50)
            summary["wall_seconds_including_startup"] = time.perf_counter() - start
            summary["host_peak_rss_mib"] = _peak_rss_mib()
            if torch.cuda.is_available():
                summary["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
                summary["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
            else:
                summary["cuda_peak_allocated_mib"] = None
                summary["cuda_peak_reserved_mib"] = None
            by_method[method] = summary
            rows.append(summary)
            print(json.dumps({"method": method, "pde": pde,
                              "step_seconds_median": summary.get("step_seconds_median"),
                              "step_seconds_p90": summary.get("step_seconds_p90"),
                              "wall_seconds": summary["wall_seconds_including_startup"],
                              "cuda_peak_allocated_mib": summary["cuda_peak_allocated_mib"],
                              "cuda_peak_reserved_mib": summary["cuda_peak_reserved_mib"],
                              "host_peak_rss_mib": summary["host_peak_rss_mib"]}), flush=True)
    # Controlled optimizer-kernel candidate: same data, seeds, models, steps and
    # loss math; only AdamW's CUDA fused implementation changes.
    for pde in ("burgers", "heat"):
        query = ROOT / "data" / "queries_v2" / "train" / pde / "teacher_queries.npz"
        for method in ("P", "B4"):
            checkpoint = out_root / f"{method}_{pde}_s11_fused_adamw.pt"
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            summary = train_operator(query, pde, method, 11, 250, 64, checkpoint,
                                     "cuda", timing_warmup=50, fused_optimizer=True)
            summary["wall_seconds_including_startup"] = time.perf_counter() - start
            summary["host_peak_rss_mib"] = _peak_rss_mib()
            summary["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            summary["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
            rows.append(summary)
            print(json.dumps({"method": method, "pde": pde, "variant": "fused_adamw",
                              "step_seconds_median": summary.get("step_seconds_median"),
                              "step_seconds_p90": summary.get("step_seconds_p90"),
                              "wall_seconds": summary["wall_seconds_including_startup"],
                              "cuda_peak_allocated_mib": summary["cuda_peak_allocated_mib"]}), flush=True)
    path = ROOT / "evidence" / "v2_runtime_pilot_50_200.json"
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

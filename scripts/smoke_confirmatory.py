"""Tiny execution-path smoke for staged shared-observer training; not scientific evidence."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pdno.training.confirmatory import (train_observer, train_operator_staged,
                                         train_direct_staged, train_b3_staged)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = ap.parse_args()
    base = ROOT / "data" / "queries_v2"
    run = ROOT / "runs" / f"smoke_confirmatory_{args.device}"
    tr, va = base / "train" / "burgers" / "teacher_queries.npz", base / "validation" / "burgers" / "teacher_queries.npz"
    observer = run / "observer.pt"
    train_observer(tr, va, "burgers", observer, seed=7, max_updates=2, eval_interval=1,
                   batch_size=2, device_name=args.device)
    p = run / "P.pt"
    p_no_rank = run / "P-no-rank.pt"
    b4 = run / "B4.pt"
    b5 = run / "B5.pt"
    b2 = run / "B2.pt"
    b3 = run / "B3.pt"
    train_operator_staged(tr, va, observer, "burgers", "P", 11, p, field_updates=2,
                          physics_updates=2, eval_interval=1, batch_size=2, device_name=args.device)
    train_operator_staged(tr, va, observer, "burgers", "P-no-rank", 11, p_no_rank, field_updates=2,
                          physics_updates=2, eval_interval=1, batch_size=2, device_name=args.device)
    train_operator_staged(tr, va, observer, "burgers", "B4", 11, b4, field_updates=2,
                          physics_updates=2, eval_interval=1, batch_size=2, device_name=args.device)
    train_operator_staged(tr, va, observer, "burgers", "B5", 11, b5, lambda_phys=0.0,
                          lambda_balance=0.0, field_updates=2, physics_updates=2,
                          eval_interval=1, batch_size=2, device_name=args.device)
    train_direct_staged(tr, va, observer, "burgers", 11, b2, max_updates=2,
                        eval_interval=1, batch_size=2, device_name=args.device)
    train_b3_staged(tr, va, b2, b4, "burgers", 11, b3, max_updates=2,
                    eval_interval=1, device_name=args.device)
    print(f"staged training smoke passed on {args.device}; outputs: {run}")


if __name__ == "__main__":
    main()

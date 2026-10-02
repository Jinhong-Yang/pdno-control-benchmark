from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdno.training.fit import train_bc, train_b3, train_operator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("P", "B4", "B5", "B2", "B3"), required=True)
    parser.add_argument("--pde", choices=("burgers", "heat"), required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timing-warmup", type=int, default=50)
    parser.add_argument("--lambda-phys", type=float, default=0.1)
    parser.add_argument("--lambda-balance", type=float, default=0.01)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--b2", type=Path)
    parser.add_argument("--b4", type=Path)
    args = parser.parse_args()
    if "train" not in args.query.parts:
        parser.error("training inputs must come from a train split")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.method in {"P", "B4", "B5"}:
        result = train_operator(args.query, args.pde, args.method, args.seed, args.steps, args.batch_size,
                                args.out, args.device, lambda_phys=args.lambda_phys,
                                lambda_balance=args.lambda_balance, timing_warmup=args.timing_warmup)
    elif args.method == "B2":
        result = train_bc(args.query, args.pde, args.seed, args.steps, args.batch_size, args.out, args.device,
                          timing_warmup=args.timing_warmup)
    else:
        if not args.b2 or not args.b4:
            parser.error("B3 requires --b2 BC checkpoint and --b4 frozen surrogate checkpoint")
        result = train_b3(args.query, args.pde, args.seed, args.steps, args.out, args.b2, args.b4, args.device,
                          timing_warmup=args.timing_warmup)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""No download, private data, paid call or cloud allocation is performed."""
import argparse
import json
import os
from pathlib import Path


def main():
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"results")
    demo.add_argument("--seed", type=int, default=53)
    demo.add_argument("--epochs", type=int, default=50)
    demo.add_argument("--per-day", type=int, default=80)
    demo.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cpu")
    report = sub.add_parser("report"); report.add_argument("--out", type=Path, required=True)
    verify = sub.add_parser("verify"); verify.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]/"results")
    args = parser.parse_args()
    if args.command == "report":
        from .report import render
        render(args.out); return
    if args.command == "verify":
        from .audit import audit
        print(json.dumps(audit(args.out), indent=2)); return
    from .experiment import run
    metrics = run(args.out, args.seed, args.epochs, args.per_day, args.encoder_device)
    print(json.dumps({"test": metrics["test"], "seconds": metrics["total_seconds"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()

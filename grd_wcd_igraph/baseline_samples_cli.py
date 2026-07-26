from __future__ import annotations

import argparse
import logging

from .baseline_samples import run_baseline_sample_sweep


def _parse_sample_sizes(value: str):
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("sample-sizes must contain at least one integer")
    sizes = []
    for part in parts:
        try:
            sizes.append(int(part))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid sample size: {part}") from exc
    return sizes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baseline-only WCD sample-size sweep (iGraph)")
    parser.add_argument(
        "--jsonl",
        action="append",
        required=True,
        help="Input generations JSONL. Repeat --jsonl to run multiple files.",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt override")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional max rows to read from each JSONL")

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name")
    parser.add_argument("--hf-token", type=str, default=None, help="HF token")

    parser.add_argument("--detector", type=str, required=True, help="Path to harm detector dill")
    parser.add_argument("--harm-threshold", type=float, default=0.75)
    parser.add_argument("--harm-start-depth", type=int, default=10)
    parser.add_argument("--harm-detector-batch-size", type=int, default=128)

    parser.add_argument(
        "--sample-sizes",
        type=_parse_sample_sizes,
        default=[100, 500, 1000, 5000, 10000],
        help="Comma-separated sample sizes, e.g. 100,500,1000,5000,10000",
    )
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for deterministic random subsets")

    parser.add_argument(
        "--report-dir",
        type=str,
        default="outputs/reports/offline_wcd_igraph_baseline_samples",
        help="Output directory for baseline-only CSV reports",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_baseline_sample_sweep(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

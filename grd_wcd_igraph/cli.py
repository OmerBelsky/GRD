from __future__ import annotations

import argparse
import logging

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone iGraph-based offline WCD runner")
    parser.add_argument("--jsonl", required=True, type=str, help="Input generations JSONL")
    parser.add_argument("--prompt", type=str, default=None, help="Optional prompt override")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional max rows to read from JSONL")

    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct", help="Model name")
    parser.add_argument("--hf-token", type=str, default=None, help="HF token")
    parser.add_argument("--model-device", type=str, default=None, help="Model device override")
    parser.add_argument("--model-backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--vllm-endpoint", type=str, default="http://127.0.0.1:8000/inference/v1/generate")
    parser.add_argument("--vllm-concurrency", type=int, default=8)

    parser.add_argument("--detector", type=str, required=True, help="Path to harm detector dill")
    parser.add_argument("--harm-threshold", type=float, default=0.75)
    parser.add_argument("--harm-start-depth", type=int, default=10)
    parser.add_argument("--harm-detector-batch-size", type=int, default=128)

    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)

    parser.add_argument("--intervention", choices=["none", "fixed_k"], default="none")
    parser.add_argument("--intervention-k", type=int, default=10)
    parser.add_argument("--intervention-top-n", type=int, default=1)
    parser.add_argument("--intervention-seed", type=int, default=0)
    parser.add_argument(
        "--intervention-selection",
        choices=["extreme", "both_sides", "min", "max", "random"],
        default="extreme",
    )
    parser.add_argument("--batch-specs-file", type=str, default=None)

    parser.add_argument("--report-dir", type=str, default="outputs/reports/offline_wcd_igraph")
    parser.add_argument("--report-filename", type=str, default=None)
    parser.add_argument("--graph-dir", type=str, default="outputs/graphs/offline_wcd_igraph")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_pipeline(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from utils.modeling import load_model_auto


@dataclass
class UniqueGeneration:
    prompt: str
    generated: str
    prompt_ids: List[int]
    continuation_ids: List[int]
    count: int = 1
    used_source: str = "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate covered probability mass from a generations JSONL file by "
            "recomputing per-token probabilities with a causal LM."
        )
    )
    parser.add_argument("--jsonl", type=str, required=True, help="Path to input generations JSONL.")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model identifier used to score generations.",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature used in generation.")
    parser.add_argument("--top-p", type=float, default=0.8, help="Nucleus top-p used in generation.")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k used in generation (0 disables).")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token (or HF_TOKEN env var).")
    parser.add_argument(
        "--model-device",
        type=str,
        default=None,
        help=(
            "Force model to a specific device (e.g. cpu, cuda, cuda:0). "
            "If unset, loader falls back to GRD_MODEL_DEVICE env var or device_map behavior."
        ),
    )
    parser.add_argument(
        "--single-free-gpu",
        action="store_true",
        help="Place model on exactly one CUDA device with the most currently free VRAM.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows loaded from JSONL for faster experimentation.",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=10,
        help="How many highest-probability unique generations to include in the JSON report.",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=None,
        help="Optional path to write a JSON report with aggregate and top-sequence stats.",
    )
    return parser.parse_args()


def pick_gpu_with_most_free_vram() -> Optional[str]:
    if not torch.cuda.is_available():
        return None

    best_idx = None
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(idx):
                free_bytes, _ = torch.cuda.mem_get_info()
        except RuntimeError:
            continue
        if free_bytes > best_free:
            best_free = int(free_bytes)
            best_idx = idx

    if best_idx is None:
        return None
    return f"cuda:{best_idx}"


def resolve_model_device_arg(args: argparse.Namespace) -> Optional[str]:
    if args.single_free_gpu:
        chosen = pick_gpu_with_most_free_vram()
        if chosen is None:
            raise SystemExit("--single-free-gpu was set but no CUDA device is available.")
        return chosen
    return args.model_device


def logaddexp_scalar(a: float, b: float) -> float:
    if math.isinf(a) and a < 0:
        return b
    if math.isinf(b) and b < 0:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def apply_generation_warpers(
    logits_1d: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        candidate_mask = torch.zeros_like(logits_1d, dtype=torch.bool)
        candidate_mask[int(torch.argmax(logits_1d).item())] = True
        filtered_scores = torch.full_like(logits_1d, -float("inf"))
        filtered_scores[candidate_mask] = logits_1d[candidate_mask]
        return filtered_scores, candidate_mask

    scores = logits_1d.unsqueeze(0)
    dummy_input_ids = torch.zeros((1, 1), dtype=torch.long, device=logits_1d.device)

    warpers = LogitsProcessorList()
    if temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(float(temperature)))
    if top_k > 0:
        warpers.append(TopKLogitsWarper(int(top_k), min_tokens_to_keep=1))

    clipped_top_p = float(min(max(top_p, 0.0), 1.0))
    if 0.0 < clipped_top_p < 1.0:
        warpers.append(TopPLogitsWarper(clipped_top_p, min_tokens_to_keep=1))

    warped_scores = warpers(dummy_input_ids, scores)[0] if len(warpers) > 0 else scores[0]
    candidate_mask = ~torch.isinf(warped_scores)

    if not bool(candidate_mask.any()):
        argmax_idx = int(torch.argmax(logits_1d).item())
        warped_scores = torch.full_like(logits_1d, -float("inf"))
        warped_scores[argmax_idx] = logits_1d[argmax_idx]
        candidate_mask = ~torch.isinf(warped_scores)

    return warped_scores, candidate_mask


def resolve_generation_text(row: dict) -> Optional[Tuple[str, str, str]]:
    """Resolve to (prompt, generated, full_text) triple."""
    prompt = row.get("prompt")
    generated = row.get("generated")
    full_text = row.get("full_text")

    if isinstance(prompt, str) and isinstance(full_text, str):
        if full_text.startswith(prompt):
            return prompt, full_text[len(prompt) :].lstrip(), full_text
        return prompt, full_text, full_text

    if isinstance(prompt, str) and isinstance(generated, str):
        return prompt, generated, None

    return None


def load_unique_generations(
    jsonl_path: str,
    tokenizer,
    max_rows: Optional[int],
) -> Tuple[List[UniqueGeneration], Dict[str, int]]:
    unique: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], UniqueGeneration] = {}

    stats = {
        "rows_read": 0,
        "rows_skipped_invalid": 0,
        "rows_empty_continuation": 0,
        "rows_skipped_empty_prompt": 0,
        "rows_duplicates": 0,
        "used_json_continuation_ids": 0,
        "used_full_text_tokenization": 0,
        "used_separate_tokenization": 0,
    }

    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if max_rows is not None and stats["rows_read"] >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            stats["rows_read"] += 1

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats["rows_skipped_invalid"] += 1
                continue

            resolved = resolve_generation_text(row)
            if resolved is None:
                stats["rows_skipped_invalid"] += 1
                continue

            prompt, generated, full_text = resolved
            
            # Tokenize prompt to get its length
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            
            if not prompt_ids:
                stats["rows_skipped_empty_prompt"] += 1
                continue
            
            continuation_ids_from_json = row.get("continuation_ids")

            # Best case: use continuation IDs produced at generation time (lossless).
            if (
                isinstance(continuation_ids_from_json, list)
                and continuation_ids_from_json
                and all(isinstance(t, int) for t in continuation_ids_from_json)
            ):
                continuation_ids = continuation_ids_from_json
                source = "json_continuation_ids"
                stats["used_json_continuation_ids"] += 1

            # Fallback for older files: retokenize full_text.
            elif full_text is not None:
                full_text_ids = tokenizer.encode(full_text, add_special_tokens=False)
                continuation_ids = full_text_ids[len(prompt_ids):]
                source = "full_text_tokenization"
                stats["used_full_text_tokenization"] += 1

            else:
                # Last resort for minimal rows that only include prompt/generated strings.
                continuation_ids = tokenizer.encode(generated, add_special_tokens=False)
                source = "separate_tokenization"
                stats["used_separate_tokenization"] += 1

            if not continuation_ids:
                stats["rows_empty_continuation"] += 1
                continue

            key = (tuple(prompt_ids), tuple(continuation_ids))
            if key in unique:
                unique[key].count += 1
                stats["rows_duplicates"] += 1
                continue

            unique[key] = UniqueGeneration(
                prompt=prompt,
                generated=generated,
                prompt_ids=prompt_ids,
                continuation_ids=continuation_ids,
                used_source=source,
            )

    return list(unique.values()), stats


def compute_nucleus_log_prob(
    filtered_scores: torch.Tensor,
    target_tid: int,
    candidate_mask: torch.Tensor,
    fallback_logprob: float,
) -> float:
    """Compute target log-prob under nucleus-filtered distribution.

    If the target token is excluded by nucleus/top-k filtering, fall back to its
    original model log-probability.
    """

    nucleus_scores = filtered_scores[candidate_mask]
    
    if nucleus_scores.numel() == 0:
        return -float("inf")
    
    # Compute logsumexp of nucleus tokens
    log_Z_nucleus = torch.logsumexp(nucleus_scores, dim=-1)
    
    # Log probability is: score - log_Z
    # But only if target is in nucleus set
    if candidate_mask[target_tid].item():
        target_score = filtered_scores[target_tid]
        return float(target_score.item() - log_Z_nucleus.item())
    else:
        # Keep the original-model token value for tokens outside nucleus.
        return fallback_logprob


def score_sequence_logs(
    model,
    prompt_ids: List[int],
    continuation_ids: List[int],
    temperature: float,
    top_p: float,
    top_k: int,
) -> Tuple[float, float, int]:
    seq_ids = prompt_ids + continuation_ids
    input_tensor = torch.tensor([seq_ids], dtype=torch.long, device=model.device)

    with torch.no_grad():
        logits = model(input_ids=input_tensor).logits[0]

    prompt_len = len(prompt_ids)
    log_mass_original = 0.0
    log_mass_nucleus = 0.0

    for j, target_tid in enumerate(continuation_ids):
        pos = prompt_len - 1 + j
        next_logits = logits[pos]

        if temperature <= 0:
            argmax_tid = int(torch.argmax(next_logits).item())
            if target_tid == argmax_tid:
                step_logprob_original = 0.0
                step_logprob_nucleus = 0.0
            else:
                step_logprob_original = -float("inf")
                step_logprob_nucleus = -float("inf")
        else:
            scaled_scores = next_logits / temperature
            full_log_probs = torch.log_softmax(scaled_scores, dim=-1)
            step_logprob_original = float(full_log_probs[target_tid].item())

            filtered_scores, candidate_mask = apply_generation_warpers(
                logits_1d=next_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            step_logprob_nucleus = compute_nucleus_log_prob(
                filtered_scores=filtered_scores,
                target_tid=target_tid,
                candidate_mask=candidate_mask,
                fallback_logprob=step_logprob_original,
            )

        log_mass_original += step_logprob_original
        log_mass_nucleus += step_logprob_nucleus

    return log_mass_original, log_mass_nucleus, len(continuation_ids)


def safe_exp(log_x: float) -> float:
    if math.isinf(log_x) and log_x < 0:
        return 0.0
    return float(math.exp(log_x))


def main() -> None:
    args = parse_args()
    model_device = resolve_model_device_arg(args)

    print("Loading model and tokenizer...")
    if model_device:
        print(f"Using single-device model placement: {model_device}")
    tokenizer, model = load_model_auto(args.model, args.hf_token, device=model_device)

    print("Reading JSONL and deduplicating generations...")
    unique_generations, load_stats = load_unique_generations(
        jsonl_path=args.jsonl,
        tokenizer=tokenizer,
        max_rows=args.max_rows,
    )

    if not unique_generations:
        raise SystemExit("No valid non-empty generations found after parsing/deduplication.")

    total_unique = len(unique_generations)
    print(f"Scoring {total_unique} unique generations (progress bar shows ETA and remaining generations)...")

    total_log_mass_original = -float("inf")
    total_log_mass_nucleus = -float("inf")
    total_tokens_scored = 0

    per_sequence = []

    for item in tqdm(unique_generations, desc="Scoring unique generations", unit="gen"):
        log_orig, log_nuc, tok_count = score_sequence_logs(
            model=model,
            prompt_ids=item.prompt_ids,
            continuation_ids=item.continuation_ids,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )

        total_log_mass_original = logaddexp_scalar(total_log_mass_original, log_orig)
        total_log_mass_nucleus = logaddexp_scalar(total_log_mass_nucleus, log_nuc)
        total_tokens_scored += tok_count

        per_sequence.append(
            {
                "prompt": item.prompt,
                "generated": item.generated,
                "duplicate_count": item.count,
                "continuation_token_count": tok_count,
                "logprob_original": log_orig,
                "logprob_nucleus": log_nuc,
                "prob_original": safe_exp(log_orig),
                "prob_nucleus": safe_exp(log_nuc),
            }
        )

    covered_mass_original = safe_exp(total_log_mass_original)
    covered_mass_nucleus = safe_exp(total_log_mass_nucleus)

    print("\n=== Coverage Summary ===")
    print(f"Rows read: {load_stats['rows_read']}")
    print(f"Unique generations: {total_unique}")
    print(f"Duplicates skipped: {load_stats['rows_duplicates']}")
    print(f"Invalid rows skipped: {load_stats['rows_skipped_invalid']}")
    print(f"Empty prompt rows skipped: {load_stats['rows_skipped_empty_prompt']}")
    print(f"Empty continuation rows skipped: {load_stats['rows_empty_continuation']}")
    print(f"Rows using JSON continuation_ids: {load_stats['used_json_continuation_ids']}")
    print(f"Rows using full_text retokenization: {load_stats['used_full_text_tokenization']}")
    print(f"Rows using separate retokenization: {load_stats['used_separate_tokenization']}")
    print(f"Total continuation tokens scored: {total_tokens_scored}")
    print(f"Covered mass (original): {covered_mass_original:.12e} (log={total_log_mass_original:.6f})")
    print(f"Covered mass (nucleus):  {covered_mass_nucleus:.12e} (log={total_log_mass_nucleus:.6f})")

    if args.report_json:
        report_dir = os.path.dirname(args.report_json)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        per_sequence_sorted = sorted(per_sequence, key=lambda x: x["prob_nucleus"], reverse=True)
        top_n = max(0, args.show_top)

        report = {
            "input": {
                "jsonl": args.jsonl,
                "model": args.model,
                "model_device": model_device,
                "single_free_gpu": args.single_free_gpu,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_rows": args.max_rows,
            },
            "stats": {
                **load_stats,
                "unique_generations": total_unique,
                "total_continuation_tokens_scored": total_tokens_scored,
            },
            "mass": {
                "covered_mass_original": covered_mass_original,
                "covered_mass_nucleus": covered_mass_nucleus,
                "covered_log_mass_original": total_log_mass_original,
                "covered_log_mass_nucleus": total_log_mass_nucleus,
            },
            "top_sequences_by_nucleus_prob": per_sequence_sorted[:top_n],
        }

        with open(args.report_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=True, indent=2)
        print(f"Wrote report to {args.report_json}")


if __name__ == "__main__":
    main()

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from urllib import error as urllib_error
from urllib import request as urllib_request
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

from utils.modeling import load_model_auto, load_tokenizer


@dataclass
class UniqueGeneration:
    prompt: str
    generated: str
    prompt_ids: List[int]
    continuation_ids: List[int]
    count: int = 1
    used_source: str = "unknown"


@dataclass
class VLLMGenerateClient:
    endpoint: str
    model_name: str
    timeout_s: float = 600.0
    vocab_token_ids: Optional[List[int]] = None

    def score_sequence(
        self,
        prompt_ids: List[int],
        continuation_ids: List[int],
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Tuple[float, float, int]:
        if self.vocab_token_ids is None:
            raise ValueError("vocab_token_ids must be provided for vLLM scoring")

        log_mass_original = 0.0
        log_mass_nucleus = 0.0

        for j, target_tid in enumerate(continuation_ids):
            prefix_ids = prompt_ids + continuation_ids[:j]
            request_payload = {
                "model": self.model_name,
                "token_ids": prefix_ids,
                "sampling_params": {
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "logprobs": 1,
                    "logprob_token_ids": self.vocab_token_ids,
                    "detokenize": False,
                },
                "stream": False,
            }
            response = self._post_json(request_payload)
            position_map = self._extract_logprob_map(response)

            step_logprob_original = position_map.get(target_tid, -float("inf"))
            step_logprob_nucleus = _compute_nucleus_log_prob_from_logprobs(
                position_map=position_map,
                target_tid=target_tid,
                top_p=top_p,
                top_k=top_k,
            )

            log_mass_original += step_logprob_original
            log_mass_nucleus += step_logprob_nucleus

        return log_mass_original, log_mass_nucleus, len(continuation_ids)

    def _extract_logprob_map(self, response: dict) -> Dict[int, float]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("vLLM response did not include choices")

        choice = choices[0]
        logprobs = choice.get("logprobs")
        if isinstance(logprobs, dict):
            content = logprobs.get("content")
            if isinstance(content, list) and content:
                position_map = self._extract_from_logprobs_content(content[0])
                if position_map:
                    return position_map

        if isinstance(logprobs, list) and logprobs:
            position_map = _normalize_prompt_logprob_map(logprobs[0])
            if position_map:
                return position_map

        raise RuntimeError("vLLM response did not include usable logprobs")

    def _extract_from_logprobs_content(self, content_item) -> Dict[int, float]:
        if not isinstance(content_item, dict):
            return {}

        position_map: Dict[int, float] = {}
        top_logprobs = content_item.get("top_logprobs")
        if isinstance(top_logprobs, list):
            for entry in top_logprobs:
                if not isinstance(entry, dict):
                    continue
                token = entry.get("token")
                logprob = entry.get("logprob")
                token_id = _parse_token_id(token)
                if token_id is not None and isinstance(logprob, (int, float)):
                    position_map[token_id] = float(logprob)

        if position_map:
            return position_map

        token = content_item.get("token")
        logprob = content_item.get("logprob")
        token_id = _parse_token_id(token)
        if token_id is not None and isinstance(logprob, (int, float)):
            position_map[token_id] = float(logprob)
        return position_map

    def _post_json(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request failed ({exc.code}): {message}") from exc


def _normalize_prompt_logprob_map(raw_entry) -> Dict[int, float]:
    if raw_entry is None:
        return {}
    if isinstance(raw_entry, dict):
        normalized: Dict[int, float] = {}
        for key, value in raw_entry.items():
            token_id = int(key)
            if isinstance(value, dict):
                normalized[token_id] = float(value["logprob"])
            else:
                normalized[token_id] = float(value)
        return normalized
    return {}


def _parse_token_id(token) -> Optional[int]:
    if not isinstance(token, str):
        return None
    if token.startswith("token_id:"):
        try:
            return int(token.split(":", 1)[1])
        except ValueError:
            return None
    try:
        return int(token)
    except ValueError:
        return None


def _get_vocab_token_ids(tokenizer) -> List[int]:
    return list(range(len(tokenizer)))


def _compute_nucleus_log_prob_from_logprobs(
    *,
    position_map: Dict[int, float],
    target_tid: int,
    top_p: float,
    top_k: int,
) -> float:
    if target_tid not in position_map:
        return -float("inf")

    sorted_items = sorted(position_map.items(), key=lambda item: item[1], reverse=True)
    candidate_items = sorted_items

    if top_k > 0:
        candidate_items = candidate_items[: max(1, top_k)]

    clipped_top_p = float(min(max(top_p, 0.0), 1.0))
    if 0.0 < clipped_top_p < 1.0:
        cumulative_prob = 0.0
        nucleus_items = []
        for token_id, logprob in candidate_items:
            nucleus_items.append((token_id, logprob))
            cumulative_prob += math.exp(logprob)
            if cumulative_prob >= clipped_top_p:
                break
        candidate_items = nucleus_items

    candidate_token_ids = {token_id for token_id, _ in candidate_items}
    if target_tid not in candidate_token_ids:
        return -float("inf")

    candidate_logprobs = torch.tensor([logprob for _, logprob in candidate_items], dtype=torch.float64)
    return float(position_map[target_tid] - torch.logsumexp(candidate_logprobs, dim=0).item())


def _get_vocab_token_ids(tokenizer) -> List[int]:
    vocab_size = len(tokenizer)
    return list(range(vocab_size))


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
    parser.add_argument(
        "--model-backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Backend used to score generation mass.",
    )
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default=os.getenv("VLLM_GENERATE_ENDPOINT", "http://127.0.0.1:8000/inference/v1/generate"),
        help="vLLM token-in/token-out generate endpoint for backend=vllm.",
    )
    parser.add_argument(
        "--vllm-concurrency",
        type=int,
        default=8,
        help="Maximum number of in-flight vLLM scoring requests to run concurrently.",
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
) -> float:
    """Compute target log-prob under nucleus-filtered distribution.

    If the target token is excluded by nucleus/top-k filtering, its strict
    top-p/top-k probability is zero.
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

    return -float("inf")


def score_sequence_logs(
    model,
    prompt_ids: List[int],
    continuation_ids: List[int],
    temperature: float,
    top_p: float,
    top_k: int,
    *,
    backend: str = "transformers",
    vllm_client: Optional[VLLMGenerateClient] = None,
) -> Tuple[float, float, int]:
    if backend == "vllm":
        if vllm_client is None:
            raise ValueError("vllm backend requires vllm_client")
        return vllm_client.score_sequence(prompt_ids, continuation_ids, temperature, top_p, top_k)

    seq_ids = prompt_ids + continuation_ids
    input_tensor = torch.tensor([seq_ids], dtype=torch.long, device=model.device)

    with torch.no_grad():
        logits = model(input_ids=input_tensor, use_cache=True).logits[0]

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
            )

        log_mass_original += step_logprob_original
        log_mass_nucleus += step_logprob_nucleus

    return log_mass_original, log_mass_nucleus, len(continuation_ids)


def score_sequence_logs_batch(
    model,
    items,
    temperature: float,
    top_p: float,
    top_k: int,
    *,
    backend: str = "transformers",
    vllm_client: Optional[VLLMGenerateClient] = None,
    batch_size: int = 32,
):
    """Batch-score several prompt/continuation pairs with a single model forward pass when possible."""
    if backend == "vllm":
        if vllm_client is None:
            raise ValueError("vllm backend requires vllm_client")
        return [
            (*item, *vllm_client.score_sequence(item[1], item[2], temperature, top_p, top_k))
            if isinstance(item, tuple) and len(item) >= 3
            else (*item, *vllm_client.score_sequence(item.prompt_ids, item.continuation_ids, temperature, top_p, top_k))
            for item in items
        ]

    results = []
    effective_batch_size = max(1, batch_size)
    n_batches = max(1, math.ceil(len(items) / effective_batch_size))
    print(
        f"Using batched transformer mass scoring for {len(items)} sequences "
        f"(batch_size={effective_batch_size}, batches={n_batches})"
    )

    for start in tqdm(
        range(0, len(items), effective_batch_size),
        total=n_batches,
        desc="Batched mass scoring",
        unit="batch",
    ):
        batch = items[start : start + effective_batch_size]
        seqs = []
        for item in batch:
            if isinstance(item, tuple) and len(item) >= 3:
                prompt_ids, continuation_ids = item[1], item[2]
            else:
                prompt_ids, continuation_ids = item.prompt_ids, item.continuation_ids
            seqs.append(prompt_ids + continuation_ids)

        max_len = max(len(seq) for seq in seqs)
        batch_tensor = torch.full((len(seqs), max_len), fill_value=0, dtype=torch.long, device=model.device)
        for i, seq in enumerate(seqs):
            batch_tensor[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=model.device)

        with torch.no_grad():
            logits = model(input_ids=batch_tensor, use_cache=True).logits

        for idx, item in enumerate(batch):
            if isinstance(item, tuple) and len(item) >= 3:
                prompt_ids, continuation_ids = item[1], item[2]
                source_item = item[0]
            else:
                prompt_ids, continuation_ids = item.prompt_ids, item.continuation_ids
                source_item = item

            prompt_len = len(prompt_ids)
            log_mass_original = 0.0
            log_mass_nucleus = 0.0

            seq_logits = logits[idx]
            for j, target_tid in enumerate(continuation_ids):
                pos = prompt_len - 1 + j
                next_logits = seq_logits[pos]

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
                    )

                log_mass_original += step_logprob_original
                log_mass_nucleus += step_logprob_nucleus

            results.append((source_item, log_mass_original, log_mass_nucleus, len(continuation_ids)))

    return results


def score_unique_generations(
    unique_generations: List[UniqueGeneration],
    *,
    model,
    backend: str,
    vllm_client: Optional[VLLMGenerateClient],
    temperature: float,
    top_p: float,
    top_k: int,
    max_workers: int,
):
    def _score_item(item: UniqueGeneration):
        log_orig, log_nuc, tok_count = score_sequence_logs(
            model=model,
            prompt_ids=item.prompt_ids,
            continuation_ids=item.continuation_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            backend=backend,
            vllm_client=vllm_client,
        )
        return item, log_orig, log_nuc, tok_count

    if backend == "vllm" and max_workers > 1 and len(unique_generations) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_score_item, item) for item in unique_generations]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Scoring unique generations",
                unit="gen",
            ):
                yield future.result()
    elif backend == "transformers" and len(unique_generations) > 1:
        batch_size = max(1, min(32, len(unique_generations)))
        print(
            f"Using batched transformer mass scoring for {len(unique_generations)} unique generations "
            f"(batch_size={batch_size})"
        )
        for result in tqdm(
            score_sequence_logs_batch(
                model=model,
                items=unique_generations,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                backend=backend,
                vllm_client=vllm_client,
                batch_size=batch_size,
            ),
            desc="Scoring unique generations (batched)",
            unit="gen",
            total=len(unique_generations),
        ):
            yield result
    else:
        for item in tqdm(unique_generations, desc="Scoring unique generations", unit="gen"):
            yield _score_item(item)


def safe_exp(log_x: float) -> float:
    if math.isinf(log_x) and log_x < 0:
        return 0.0
    return float(math.exp(log_x))


def main() -> None:
    args = parse_args()
    model_device = resolve_model_device_arg(args)

    print("Loading model and tokenizer...")
    if args.model_backend == "vllm":
        tokenizer = load_tokenizer(args.model, args.hf_token)
        model = None
        vllm_client = VLLMGenerateClient(
            endpoint=args.vllm_endpoint,
            model_name=args.model,
            vocab_token_ids=_get_vocab_token_ids(tokenizer),
        )
        print(f"Using vLLM generate endpoint: {args.vllm_endpoint}")
    else:
        if model_device:
            print(f"Using single-device model placement: {model_device}")
        tokenizer, model = load_model_auto(args.model, args.hf_token, device=model_device)
        vllm_client = None

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

    for item, log_orig, log_nuc, tok_count in score_unique_generations(
        unique_generations,
        model=model,
        backend=args.model_backend,
        vllm_client=vllm_client,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_workers=max(1, int(args.vllm_concurrency)),
    ):

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
                "model_backend": args.model_backend,
                "vllm_endpoint": args.vllm_endpoint if args.model_backend == "vllm" else None,
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

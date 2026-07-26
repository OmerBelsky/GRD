from __future__ import annotations

import csv
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .graph_builder import _extract_continuation_ids, append_continuations_to_prefix_graph, create_empty_prefix_graph
from .harm_labeling import assign_goal_labels, score_harm_probabilities_for_nodes
from .propagation import compute_reachable_nodes, compute_wcd, propagate_labels


@dataclass
class BaselineSweepRow:
    sample_size_requested: int
    sample_size_used: int
    baseline_wcd: int
    witness_node: Optional[int]
    harmful_goal_count: int
    safe_goal_count: int
    runtime_seconds: float
    seed: int


def _normalize_sample_sizes(sample_sizes: Sequence[int]) -> List[int]:
    normalized: List[int] = []
    for value in sample_sizes:
        int_value = int(value)
        if int_value <= 0:
            raise ValueError(f"Sample sizes must be positive integers. Received: {int_value}")
        normalized.append(int_value)
    return sorted(set(normalized))


def _load_valid_continuations(
    *,
    jsonl_path: str,
    tokenizer,
    prompt_override: Optional[str],
    max_rows: Optional[int],
) -> Tuple[str, List[List[int]], Dict[str, int]]:
    prompt_text: Optional[str] = None
    continuations: List[List[int]] = []

    rows_read = 0
    rows_skipped_invalid = 0
    rows_empty_continuation = 0
    rows_skipped_empty_prompt = 0

    with open(jsonl_path, "r", encoding="utf-8") as file_handle:
        row_idx = 0
        for raw in file_handle:
            if max_rows is not None and row_idx >= max_rows:
                break
            line = raw.strip()
            if not line:
                continue

            row = json.loads(line)
            row_idx += 1
            rows_read += 1

            row_prompt = row.get("prompt")
            if not isinstance(row_prompt, str) or not row_prompt.strip():
                rows_skipped_empty_prompt += 1
                continue

            if prompt_text is None:
                prompt_text = row_prompt

            active_prompt = prompt_override or prompt_text
            prompt_tokens = tokenizer.encode(active_prompt, add_special_tokens=False)
            continuation_ids = _extract_continuation_ids(row, tokenizer, prompt_tokens)
            if not continuation_ids:
                rows_empty_continuation += 1
                continue

            if not isinstance(continuation_ids, list) or not all(isinstance(tok, int) for tok in continuation_ids):
                rows_skipped_invalid += 1
                continue

            continuations.append(continuation_ids)

    if prompt_text is None:
        raise ValueError(f"No valid prompt rows found in JSONL: {jsonl_path}")

    stats = {
        "rows_read": rows_read,
        "rows_skipped_invalid": rows_skipped_invalid,
        "rows_empty_continuation": rows_empty_continuation,
        "rows_skipped_empty_prompt": rows_skipped_empty_prompt,
        "valid_rows": len(continuations),
    }
    return (prompt_override or prompt_text), continuations, stats


def _build_output_path(report_dir: str, jsonl_path: str) -> str:
    stem = Path(jsonl_path).stem
    filename = f"baseline_wcd_samples__{stem}.csv"
    return str(Path(report_dir) / filename)


def _write_csv(output_path: str, jsonl_path: str, rows: List[BaselineSweepRow]) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as file_handle:
        writer = csv.writer(file_handle)
        writer.writerow(
            [
                "jsonl_file",
                "sample_size_requested",
                "sample_size_used",
                "baseline_wcd",
                "witness_node",
                "harmful_goal_count",
                "safe_goal_count",
                "runtime_seconds",
                "seed",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    os.path.basename(jsonl_path),
                    row.sample_size_requested,
                    row.sample_size_used,
                    row.baseline_wcd,
                    "" if row.witness_node is None else row.witness_node,
                    row.harmful_goal_count,
                    row.safe_goal_count,
                    f"{row.runtime_seconds:.6f}",
                    row.seed,
                ]
            )
    return output_path


def _run_single_file(args, *, jsonl_path: str, sample_sizes: List[int]) -> str:
    logger = logging.getLogger("grd_wcd_igraph.baseline_samples")

    from utils.harm import load_harm_detector
    from utils.modeling import load_tokenizer

    tokenizer = load_tokenizer(args.model, hf_token=args.hf_token)
    harm_detector = load_harm_detector(args.detector)

    prompt_text, continuations, load_stats = _load_valid_continuations(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        prompt_override=args.prompt,
        max_rows=args.max_rows,
    )

    total_valid = len(continuations)
    if total_valid == 0:
        raise ValueError(f"No valid continuation rows found in JSONL: {jsonl_path}")

    rng = random.Random(args.sample_seed)
    shuffled_indices = list(range(total_valid))
    rng.shuffle(shuffled_indices)

    prefix_graph = create_empty_prefix_graph(prompt_text)
    harm_probs = [0.0] * len(prefix_graph.depth)

    computed_by_size: Dict[int, Tuple[int, Optional[int], int, int, float]] = {}
    rows: List[BaselineSweepRow] = []
    prev_size_used = 0
    file_start = time.perf_counter()

    for requested_size in sample_sizes:
        size_used = min(requested_size, total_valid)
        if size_used not in computed_by_size:
            new_indices = shuffled_indices[prev_size_used:size_used]
            new_continuations = [continuations[idx] for idx in new_indices]
            new_node_ids = append_continuations_to_prefix_graph(
                prefix_graph,
                new_continuations,
                row_start=prev_size_used + 1,
            )

            if len(harm_probs) < len(prefix_graph.depth):
                harm_probs.extend([0.0] * (len(prefix_graph.depth) - len(harm_probs)))

            target_new_nodes = [
                node_id
                for node_id in new_node_ids
                if node_id != 0 and prefix_graph.depth[node_id] >= args.harm_start_depth
            ]
            if target_new_nodes:
                node_scores = score_harm_probabilities_for_nodes(
                    prefix_graph=prefix_graph,
                    tokenizer=tokenizer,
                    harm_detector=harm_detector,
                    node_ids=target_new_nodes,
                    batch_size=args.harm_detector_batch_size,
                )
                for node_id, score in node_scores.items():
                    harm_probs[node_id] = score

            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            goal_labels = assign_goal_labels(
                prefix_graph=prefix_graph,
                harm_probs=harm_probs,
                harm_threshold=args.harm_threshold,
                harm_start_depth=args.harm_start_depth,
                eos_token_id=eos_token_id,
            )
            propagation = propagate_labels(
                prefix_graph=prefix_graph,
                harmful_goal=goal_labels.harmful_goal,
                safe_goal=goal_labels.safe_goal,
            )
            reachable = compute_reachable_nodes(prefix_graph=prefix_graph)
            baseline_wcd, witness_node = compute_wcd(prefix_graph, propagation, reachable=reachable)

            elapsed = time.perf_counter() - file_start
            computed_by_size[size_used] = (
                baseline_wcd,
                witness_node,
                goal_labels.harmful_goal_count,
                goal_labels.safe_goal_count,
                elapsed,
            )
            prev_size_used = size_used

        baseline_wcd, witness_node, harmful_count, safe_count, elapsed = computed_by_size[size_used]
        rows.append(
            BaselineSweepRow(
                sample_size_requested=requested_size,
                sample_size_used=size_used,
                baseline_wcd=baseline_wcd,
                witness_node=witness_node,
                harmful_goal_count=harmful_count,
                safe_goal_count=safe_count,
                runtime_seconds=elapsed,
                seed=args.sample_seed,
            )
        )

    logger.info(
        "Completed baseline sweep for %s with %d valid rows; requested sizes=%s",
        jsonl_path,
        total_valid,
        sample_sizes,
    )
    logger.debug("Load stats: %s", load_stats)

    output_path = _build_output_path(args.report_dir, jsonl_path)
    return _write_csv(output_path, jsonl_path, rows)


def run_baseline_sample_sweep(args) -> List[str]:
    sample_sizes = _normalize_sample_sizes(args.sample_sizes)
    output_paths: List[str] = []
    for jsonl_path in args.jsonl:
        output_paths.append(_run_single_file(args, jsonl_path=jsonl_path, sample_sizes=sample_sizes))
    return output_paths

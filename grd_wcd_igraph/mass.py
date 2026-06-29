from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from tqdm import tqdm

from .types import MassResult, PrefixGraph


def compute_mass_result(
    *,
    prefix_graph: PrefixGraph,
    tokenizer,
    model,
    backend: str,
    vllm_client,
    jsonl_path: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_rows: Optional[int],
    vllm_concurrency: int,
) -> MassResult:
    from grd_generation_mass import load_unique_generations, logaddexp_scalar, safe_exp, score_sequence_logs

    unique_generations, load_stats = load_unique_generations(jsonl_path=jsonl_path, tokenizer=tokenizer, max_rows=max_rows)

    node_log_mass_original = [-float("inf")] * len(prefix_graph.depth)
    node_log_mass_nucleus = [-float("inf")] * len(prefix_graph.depth)
    total_log_mass_original = -float("inf")
    total_log_mass_nucleus = -float("inf")
    total_tokens_scored = 0

    def _score_item(item):
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

    def _accumulate(item, log_orig: float, log_nuc: float, tok_count: int):
        nonlocal total_log_mass_original, total_log_mass_nucleus, total_tokens_scored
        total_log_mass_original = logaddexp_scalar(total_log_mass_original, log_orig)
        total_log_mass_nucleus = logaddexp_scalar(total_log_mass_nucleus, log_nuc)
        total_tokens_scored += tok_count

        prefix = []
        for tok in item.continuation_ids:
            prefix.append(tok)
            node_id = prefix_graph.path_to_vid.get(tuple(prefix))
            if node_id is None:
                continue
            node_log_mass_original[node_id] = logaddexp_scalar(node_log_mass_original[node_id], log_orig)
            node_log_mass_nucleus[node_id] = logaddexp_scalar(node_log_mass_nucleus[node_id], log_nuc)

    if backend == "vllm" and vllm_concurrency > 1 and len(unique_generations) > 1:
        with ThreadPoolExecutor(max_workers=vllm_concurrency) as executor:
            futures = [executor.submit(_score_item, item) for item in unique_generations]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scoring mass", unit="gen"):
                item, log_orig, log_nuc, tok_count = future.result()
                _accumulate(item, log_orig, log_nuc, tok_count)
    else:
        for item in tqdm(unique_generations, desc="Scoring mass", unit="gen"):
            item, log_orig, log_nuc, tok_count = _score_item(item)
            _accumulate(item, log_orig, log_nuc, tok_count)

    mass_summary = {
        "rows_read": load_stats["rows_read"],
        "rows_skipped_invalid": load_stats["rows_skipped_invalid"],
        "rows_empty_continuation": load_stats["rows_empty_continuation"],
        "rows_skipped_empty_prompt": load_stats["rows_skipped_empty_prompt"],
        "rows_duplicates": load_stats["rows_duplicates"],
        "unique_generations": len(unique_generations),
        "total_continuation_tokens_scored": total_tokens_scored,
        "covered_mass_original": safe_exp(total_log_mass_original),
        "covered_mass_nucleus": safe_exp(total_log_mass_nucleus),
        "covered_log_mass_original": total_log_mass_original,
        "covered_log_mass_nucleus": total_log_mass_nucleus,
    }

    return MassResult(
        mass_summary=mass_summary,
        covered_mass_nucleus=float(mass_summary["covered_mass_nucleus"]),
        covered_mass_original=float(mass_summary["covered_mass_original"]),
        node_log_mass_nucleus=node_log_mass_nucleus,
        node_log_mass_original=node_log_mass_original,
    )

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

from .graph_builder import get_node_text
from .types import GoalLabels, PrefixGraph


def _chunked(items: List[int], size: int):
    chunk_size = max(1, int(size))
    for idx in range(0, len(items), chunk_size):
        yield items[idx : idx + chunk_size]


def score_harm_probabilities(
    *,
    prefix_graph: PrefixGraph,
    tokenizer,
    harm_detector,
    harm_start_depth: int,
    batch_size: int,
) -> List[float]:
    from utils.harm import harm_proba_batch

    harm_probs = [0.0] * len(prefix_graph.depth)
    target_nodes = [vid for vid, d in enumerate(prefix_graph.depth) if d >= harm_start_depth and vid != 0]
    if not target_nodes:
        return harm_probs

    for batch in tqdm(list(_chunked(target_nodes, batch_size)), desc="Scoring harm", unit="batch"):
        texts = [get_node_text(tokenizer, prefix_graph, vid) for vid in batch]
        probs = harm_proba_batch(harm_detector, texts)
        for vid, prob in zip(batch, probs):
            harm_probs[vid] = float(prob)

    return harm_probs


def assign_goal_labels(
    *,
    prefix_graph: PrefixGraph,
    harm_probs: List[float],
    harm_threshold: float,
    harm_start_depth: int,
    eos_token_id: Optional[int],
) -> GoalLabels:
    node_count = len(prefix_graph.depth)
    harmful_goal = [False] * node_count
    safe_goal = [False] * node_count

    harmful_count = 0
    safe_count = 0

    for leaf_id in prefix_graph.leaves:
        ancestry: List[int] = []
        cursor = leaf_id
        while cursor > 0:
            ancestry.append(cursor)
            cursor = prefix_graph.parent[cursor]
        ancestry.reverse()

        first_harmful_prefix: Optional[int] = None
        for node_id in ancestry:
            if prefix_graph.depth[node_id] < harm_start_depth:
                continue
            if harm_probs[node_id] > harm_threshold:
                first_harmful_prefix = node_id
                break

        if first_harmful_prefix is not None:
            if not harmful_goal[first_harmful_prefix]:
                harmful_goal[first_harmful_prefix] = True
                harmful_count += 1
            continue

        depth = prefix_graph.depth[leaf_id]
        last_tok = prefix_graph.token_id[leaf_id]
        reached_terminal = depth == prefix_graph.max_depth or (eos_token_id is not None and last_tok == eos_token_id)
        if reached_terminal and not safe_goal[leaf_id]:
            safe_goal[leaf_id] = True
            safe_count += 1

    return GoalLabels(
        harmful_goal=harmful_goal,
        safe_goal=safe_goal,
        harmful_goal_count=harmful_count,
        safe_goal_count=safe_count,
    )


def probs_for_nodes(node_ids: List[int], harm_probs: List[float]) -> Dict[int, float]:
    return {node_id: float(harm_probs[node_id]) for node_id in node_ids}

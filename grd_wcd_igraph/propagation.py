from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .types import PrefixGraph, PropagationResult


def propagate_labels(
    *,
    prefix_graph: PrefixGraph,
    harmful_goal: List[bool],
    safe_goal: List[bool],
    allowed_children_by_parent: Optional[Dict[int, Set[int]]] = None,
    intervention_depth: Optional[int] = None,
    baseline: Optional[PropagationResult] = None,
) -> PropagationResult:
    node_count = len(prefix_graph.depth)

    if baseline is not None and intervention_depth is not None:
        reach_harmful = list(baseline.reach_harmful)
        reach_safe = list(baseline.reach_safe)
        start_depth = max(0, intervention_depth - 1)
    else:
        reach_harmful = [False] * node_count
        reach_safe = [False] * node_count
        start_depth = prefix_graph.max_depth

    for depth in range(start_depth, -1, -1):
        for node_id in prefix_graph.nodes_by_depth.get(depth, []):
            child_ids = prefix_graph.children[node_id]
            if intervention_depth is not None and depth == max(0, intervention_depth - 1) and allowed_children_by_parent is not None:
                allowed = allowed_children_by_parent.get(node_id)
                if allowed is not None:
                    child_ids = [child for child in child_ids if child in allowed]

            child_harmful = any(reach_harmful[child_id] for child_id in child_ids)
            child_safe = any(reach_safe[child_id] for child_id in child_ids)
            reach_harmful[node_id] = harmful_goal[node_id] or child_harmful
            reach_safe[node_id] = safe_goal[node_id] or child_safe

    ambiguous = [reach_harmful[idx] and reach_safe[idx] for idx in range(node_count)]
    return PropagationResult(reach_harmful=reach_harmful, reach_safe=reach_safe, ambiguous=ambiguous)


def compute_reachable_nodes(
    *,
    prefix_graph: PrefixGraph,
    allowed_children_by_parent: Optional[Dict[int, Set[int]]] = None,
    intervention_depth: Optional[int] = None,
) -> List[bool]:
    reachable = [False] * len(prefix_graph.depth)
    queue: deque[int] = deque([0])
    reachable[0] = True

    while queue:
        node_id = queue.popleft()
        child_ids = prefix_graph.children[node_id]

        if (
            intervention_depth is not None
            and prefix_graph.depth[node_id] == max(0, intervention_depth - 1)
            and allowed_children_by_parent is not None
            and node_id in allowed_children_by_parent
        ):
            allowed = allowed_children_by_parent[node_id]
            child_ids = [child for child in child_ids if child in allowed]

        for child_id in child_ids:
            if reachable[child_id]:
                continue
            reachable[child_id] = True
            queue.append(child_id)

    return reachable


def compute_wcd(prefix_graph: PrefixGraph, propagation: PropagationResult, reachable: Optional[List[bool]] = None) -> Tuple[int, Optional[int]]:
    best_depth = -1
    best_node: Optional[int] = None
    for node_id, is_ambiguous in enumerate(propagation.ambiguous):
        if not is_ambiguous:
            continue
        if reachable is not None and not reachable[node_id]:
            continue
        depth = prefix_graph.depth[node_id]
        if depth > best_depth:
            best_depth = depth
            best_node = node_id

    return best_depth, best_node

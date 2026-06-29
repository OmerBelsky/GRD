from __future__ import annotations

import json
import random
from typing import Dict, List, Optional, Set, Tuple

from .types import InterventionEvent, InterventionSelection, InterventionSpec, PrefixGraph


def load_batch_specs(batch_specs_file: str) -> List[InterventionSpec]:
    with open(batch_specs_file, "r", encoding="utf-8") as file_handle:
        specs = json.load(file_handle)
    if not isinstance(specs, list):
        raise ValueError("Batch specs file must contain a JSON list.")

    normalized: List[InterventionSpec] = []
    for raw in specs:
        if not isinstance(raw, dict):
            raise ValueError("Each spec must be a JSON object.")
        normalized.append(
            InterventionSpec(
                intervention=str(raw.get("intervention", "none")),
                intervention_k=raw.get("intervention_k"),
                intervention_top_n=int(raw.get("intervention_top_n", 1)),
                intervention_selection=str(raw.get("intervention_selection", "extreme")),
                intervention_seed=raw.get("intervention_seed"),
            )
        )
    return normalized


def spec_from_args(args) -> InterventionSpec:
    return InterventionSpec(
        intervention=args.intervention,
        intervention_k=args.intervention_k,
        intervention_top_n=args.intervention_top_n,
        intervention_selection=args.intervention_selection,
        intervention_seed=args.intervention_seed,
    )


def _select_children(
    scored_children: List[Tuple[int, float]],
    mode: str,
    top_n: int,
    rng: Optional[random.Random] = None,
) -> List[int]:
    n = max(1, int(top_n))
    if mode == "extreme":
        ranked = sorted(scored_children, key=lambda item: (abs(item[1] - 0.5), item[1]), reverse=True)
        return [child_id for child_id, _ in ranked[:n]]

    if mode == "max":
        ranked = sorted(scored_children, key=lambda item: (item[1], -item[0]), reverse=True)
        return [child_id for child_id, _ in ranked[:n]]

    if mode == "min":
        ranked = sorted(scored_children, key=lambda item: (item[1], item[0]))
        return [child_id for child_id, _ in ranked[:n]]

    if mode == "both_sides":
        high = sorted(scored_children, key=lambda item: (item[1], -item[0]), reverse=True)
        low = sorted(scored_children, key=lambda item: (item[1], item[0]))
        selected: List[int] = []
        for child_id, _ in high[:n]:
            if child_id not in selected:
                selected.append(child_id)
        for child_id, _ in low[:n]:
            if child_id not in selected:
                selected.append(child_id)
        return selected

    if mode == "random":
        child_ids = [child_id for child_id, _ in scored_children]
        if n >= len(child_ids):
            return child_ids
        random_source = rng if rng is not None else random
        return random_source.sample(child_ids, k=n)

    raise ValueError(f"Unknown intervention selection mode: {mode}")


def build_intervention_selection(
    *,
    prefix_graph: PrefixGraph,
    spec: InterventionSpec,
    harm_probs: List[float],
) -> InterventionSelection:
    if spec.intervention == "none":
        return InterventionSelection(allowed_children_by_parent={}, events=[])
    if spec.intervention != "fixed_k":
        raise ValueError(f"Unsupported intervention: {spec.intervention}")
    if spec.intervention_k is None or spec.intervention_k <= 0:
        return InterventionSelection(allowed_children_by_parent={}, events=[])

    trigger_depth = spec.intervention_k - 1
    rng = random.Random(spec.intervention_seed) if spec.intervention_seed is not None else None
    allowed_children_by_parent: Dict[int, Set[int]] = {}
    events: List[InterventionEvent] = []

    for parent_id in prefix_graph.nodes_by_depth.get(trigger_depth, []):
        child_ids = prefix_graph.children[parent_id]
        if not child_ids:
            continue

        scored = [(child_id, float(harm_probs[child_id])) for child_id in child_ids]
        selected = _select_children(
            scored,
            spec.intervention_selection,
            spec.intervention_top_n,
            rng=rng,
        )
        allowed_children_by_parent[parent_id] = set(selected)

        chosen = selected[0] if selected else None
        chosen_prob = None if chosen is None else float(harm_probs[chosen])
        events.append(
            InterventionEvent(
                parent_id=parent_id,
                depth=spec.intervention_k,
                candidate_child_ids=list(child_ids),
                selected_child_ids=list(selected),
                chosen_child_id=chosen,
                chosen_harm_probability=chosen_prob,
            )
        )

    return InterventionSelection(allowed_children_by_parent=allowed_children_by_parent, events=events)

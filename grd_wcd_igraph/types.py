from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .igraph_compat import Graph


@dataclass
class PrefixGraph:
    graph: Graph
    prompt: str
    max_depth: int
    parent: List[int]
    depth: List[int]
    token_id: List[Optional[int]]
    gen_tokens: List[Tuple[int, ...]]
    children: List[List[int]]
    nodes_by_depth: Dict[int, List[int]]
    leaves: List[int]
    path_to_vid: Dict[Tuple[int, ...], int]
    visit_count: List[int]
    first_seen_row: List[int]


@dataclass
class GoalLabels:
    harmful_goal: List[bool]
    safe_goal: List[bool]
    harmful_goal_count: int
    safe_goal_count: int


@dataclass
class PropagationResult:
    reach_harmful: List[bool]
    reach_safe: List[bool]
    ambiguous: List[bool]


@dataclass
class InterventionSpec:
    intervention: str
    intervention_k: Optional[int] = None
    intervention_top_n: int = 1
    intervention_selection: str = "extreme"
    intervention_seed: Optional[int] = None


@dataclass
class InterventionEvent:
    parent_id: int
    depth: int
    candidate_child_ids: List[int]
    selected_child_ids: List[int]
    chosen_child_id: Optional[int]
    chosen_harm_probability: Optional[float]


@dataclass
class InterventionSelection:
    allowed_children_by_parent: Dict[int, Set[int]]
    events: List[InterventionEvent]


@dataclass
class MassResult:
    mass_summary: Dict[str, object]
    covered_mass_nucleus: float
    covered_mass_original: float
    node_log_mass_nucleus: List[float]
    node_log_mass_original: List[float]

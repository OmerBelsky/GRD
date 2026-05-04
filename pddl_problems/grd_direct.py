from __future__ import annotations

import json
import time
import os
from array import array
import heapq
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.harm import is_harmful_cached, is_harmful_text, load_harm_detector


@torch.no_grad()
def nucleus_tokens(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    prefix: str,
    p: float,
    temperature: float,
    device: str,
    top_k: int = 1,
) -> Iterable[int]:
    """
    Return token ids inside the nucleus set. To keep memory usage low, all heavy
    ops happen on CPU and gradients/caches are disabled.
    """
    enc = tokenizer(prefix, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False)
        logits = (out.logits[0, -1].float().div(temperature)).cpu()
        del out

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    sorted_logprobs = torch.log_softmax(sorted_logits, dim=-1)
    sorted_probs = sorted_logprobs.exp()
    cum_probs = torch.cumsum(sorted_probs, dim=-1)

    cutoff = int(torch.searchsorted(cum_probs, p).item())
    top_p = sorted_indices[: cutoff + 1]

    if top_p.numel() > 2 * top_k:
        sampled = torch.cat([top_p[:top_k], top_p[-top_k:]])
    else:
        sampled = top_p

    for idx in sampled.tolist():
        yield idx


@dataclass
class SearchState:
    g0_id: int
    g1_id: int
    split: bool
    done0: bool
    found_harmful0: bool
    found_no_harm1: bool
    harmful1_streak: int
    depth: int
    gen_steps: int
    total_cost: int
    parent: Optional[int]
    action: Optional[str]
    params: Tuple


@dataclass
class SearchResult:
    plan: Optional[List[Tuple[str, Tuple]]]
    expanded: int
    timed_out: bool
    best_cost: Optional[int]


class GRDDirectSearch:
    """
    Lightweight GRD planner that mirrors the PDDLStream formulation but performs
    search directly without building PDDL artifacts. The planner keeps a compact
    token tree and streams search events to a JSONL log for later visualization.
    """

    def _init_node_storage(self) -> None:
        self.NULL_TOKEN = -1
        self.NULL_PARENT = -1
        self.tokens = array("i")
        self.parents = array("i")

    def _add_node(self, token: Optional[int], parent: Optional[int]) -> int:
        node_id = len(self.tokens)
        self.tokens.append(self.NULL_TOKEN if token is None else int(token))
        self.parents.append(self.NULL_PARENT if parent is None else int(parent))
        return node_id

    def _tokens_from_node(self, node_id: int) -> List[int]:
        tokens: List[int] = []
        cursor: int = node_id
        while cursor != self.NULL_PARENT:
            tok = self.tokens[cursor]
            if tok != self.NULL_TOKEN:
                tokens.append(tok)
            cursor = self.parents[cursor]
        tokens.reverse()
        return tokens

    def _parse_state(self, state: str) -> Tuple[int, int]:
        g0_part, g1_part = state.split("<GEN_SEP>")
        g0_id = int(g0_part.split("_", 1)[1]) if "_" in g0_part else 0
        g1_id = int(g1_part.split("_", 1)[1]) if "_" in g1_part else 0
        return g0_id, g1_id

    def _build_state(self, g0_id: int, g1_id: int) -> str:
        return f"g0_{g0_id}<GEN_SEP>g1_{g1_id}"

    def _get_agent_node_id(self, state: str, agent: str) -> int:
        g0_id, g1_id = self._parse_state(state)
        return g1_id if agent == "1" else g0_id

    def tokens_for_state(self, state: str, agent: str) -> List[int]:
        return self._tokens_from_node(self._get_agent_node_id(state, agent))

    def _decode_from_node(self, node_id: int) -> str:
        tokens = self._tokens_from_node(node_id)
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

    def _decoded_generated(self, node_id: int) -> str:
        tokens = self._tokens_from_node(node_id)[self.input_length :]
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

    def __init__(
        self,
        start_text: str,
        model,
        tokenizer,
        device,
        nucleus_p: float = 0.9,
        temperature: float = 0.55,
        harm_detector_path: str = "harm_detector/models/binary_harm_detector.dill",
        max_length: Optional[int] = None,
        max_nodes: Optional[int] = None,
        max_frontier_per_depth: Optional[int] = None,
        max_depth: Optional[int] = None,
        tree_log_path: Optional[str] = "outputs/search_trees/direct/grd_search_tree.jsonl",
        action_costs: Optional[dict[str, int]] = None,
        progress_every_depth: Optional[int] = None,
        harmful_streak_prune: int = 5,
        checkpoint: Optional[dict] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        self.harm_detector = load_harm_detector(harm_detector_path)

        self.tree_log = None
        self.resume_frontier: Optional[list[Tuple[int, int, int]]] = None
        self.resume_best_goal_cost: Optional[int] = None
        self.resume_best_goal_state_id: Optional[int] = None
        self.resume_tie: int = 0

        if checkpoint is not None:
            cfg = checkpoint["config"]
            self.start_text = cfg["start_text"]
            self.nucleus_p = cfg["nucleus_p"]
            self.temperature = cfg["temperature"]
            self.max_length = cfg["max_length"]
            self.max_nodes = cfg["max_nodes"]
            self.max_frontier_per_depth = cfg["max_frontier_per_depth"]
            self.max_depth = cfg["max_depth"]
            self.tree_log_path = tree_log_path or cfg.get("tree_log_path")
            self.frontier_counts = checkpoint.get("frontier_counts", {})
            self.action_costs = cfg.get(
                "action_costs",
                {"generate-01": 95, "generate-0": 100, "generate-1": 100},
            )
            self.progress_every_depth = cfg.get("progress_every_depth")
            self.harmful_streak_prune = cfg.get("harmful_streak_prune", harmful_streak_prune)
            self._harmful_cache: dict[int, bool] = {}

            self.tokens = array("i", checkpoint["tokens"])
            self.parents = array("i", checkpoint["parents"])
            self.input_length = checkpoint["input_length"]
            self.root_node_id = checkpoint["root_node_id"]
            self.start_node_id = checkpoint["start_node_id"]
            self.start_obj = checkpoint["start_obj"]
            self.generated_states = set(checkpoint["generated_states"])
            self.best_cost = {
                tuple(key[:6]): key[6] for key in checkpoint.get("best_cost", [])
            }
            missing_harmful_streak = False
            loaded_states = []
            for s in checkpoint["states"]:
                if "harmful1_streak" not in s:
                    missing_harmful_streak = True
                loaded_states.append(
                    SearchState(
                        g0_id=s["g0_id"],
                        g1_id=s["g1_id"],
                        split=s["split"],
                        done0=s["done0"],
                        found_harmful0=s["found_harmful0"],
                        found_no_harm1=s["found_no_harm1"],
                        harmful1_streak=s.get("harmful1_streak", 0),
                        depth=s["depth"],
                        gen_steps=s["gen_steps"],
                        total_cost=s["total_cost"],
                        parent=s["parent"],
                        action=s["action"],
                        params=tuple(s["params"]),
                    )
                )
            self.states = loaded_states
            self.reported_depths = set(checkpoint.get("reported_depths", []))
            self.first_split_reported = checkpoint.get("first_split_reported", False)
            self.gen_steps_at_first_split = checkpoint.get("gen_steps_at_first_split")
            self.first_done0_reported = checkpoint.get("first_done0_reported", False)
            self.resume_frontier = [tuple(item) for item in checkpoint.get("frontier", [])]
            self.resume_best_goal_cost = checkpoint.get("best_goal_cost")
            self.resume_best_goal_state_id = checkpoint.get("best_goal_state_id")
            self.resume_tie = checkpoint.get("tie", 0)
            self.nodes_per_depth = checkpoint.get("nodes_per_depth", {})
            self.expanded_per_depth = checkpoint.get("expanded_per_depth", {})
            if not self.nodes_per_depth:
                self.nodes_per_depth = {}
                for s in self.states:
                    self.nodes_per_depth[s.depth] = self.nodes_per_depth.get(s.depth, 0) + 1
            if not self.expanded_per_depth:
                self.expanded_per_depth = {}
            if missing_harmful_streak:
                self._refresh_harmful1_streaks()
        else:
            self.start_text = start_text
            self.nucleus_p = nucleus_p
            self.temperature = temperature
            self.max_length = max_length
            self.max_nodes = max_nodes
            self.max_frontier_per_depth = max_frontier_per_depth
            self.max_depth = max_depth
            self.tree_log_path = tree_log_path
            self.frontier_counts: dict[int, int] = {}
            self.action_costs = action_costs or {
                "generate-01": 95,
                "generate-0": 100,
                "generate-1": 100,
            }
            self.progress_every_depth = progress_every_depth
            self.harmful_streak_prune = harmful_streak_prune
            self._harmful_cache: dict[int, bool] = {}

            self._init_node_storage()

            try:
                start_tokens: List[int] = self.tokenizer.encode(start_text, add_special_tokens=False)
            except Exception:
                start_tokens = []

            self.input_length = len(start_tokens)
            if self.max_length is not None:
                self.max_length += len(start_tokens)

            self.root_node_id = self._add_node(token=None, parent=None)
            current_node = self.root_node_id
            for tok in start_tokens:
                current_node = self._add_node(token=tok, parent=current_node)
            self.start_node_id = current_node
            self.start_obj = self._build_state(self.start_node_id, self.start_node_id)

            self.generated_states: Set[str] = {self.start_obj}
            self.best_cost: dict[Tuple[int, int, bool, bool, bool, bool], int] = {}
            self.states: List[SearchState] = []
            self.reported_depths: Set[int] = set()
            self.first_split_reported = False
            self.gen_steps_at_first_split = None
            self.first_done0_reported = False
            self.nodes_per_depth: dict[int, int] = {}
            self.expanded_per_depth: dict[int, int] = {}

    def _state_key(
        self,
        g0_id: int,
        g1_id: int,
        split: bool,
        done0: bool,
        found_harmful0: bool,
        found_no_harm1: bool,
    ) -> Tuple[int, int, bool, bool, bool, bool]:
        return (g0_id, g1_id, split, done0, found_harmful0, found_no_harm1)

    def _make_checkpoint(
        self,
        frontier: list[Tuple[int, int, int]],
        best_goal_cost: Optional[int],
        best_goal_state_id: Optional[int],
        tie: int,
    ) -> dict:
        return {
            "config": {
                "start_text": self.start_text,
                "nucleus_p": self.nucleus_p,
                "temperature": self.temperature,
                "max_length": self.max_length,
                "max_nodes": self.max_nodes,
                "max_frontier_per_depth": self.max_frontier_per_depth,
                "max_depth": self.max_depth,
                "tree_log_path": self.tree_log_path,
                "action_costs": self.action_costs,
                "progress_every_depth": self.progress_every_depth,
                "harmful_streak_prune": self.harmful_streak_prune,
            },
            "tokens": list(self.tokens),
            "parents": list(self.parents),
            "input_length": self.input_length,
            "root_node_id": self.root_node_id,
            "start_node_id": self.start_node_id,
            "start_obj": self.start_obj,
            "generated_states": list(self.generated_states),
            "best_cost": [list(key) + [cost] for key, cost in self.best_cost.items()],
            "states": [
                {
                    "g0_id": s.g0_id,
                    "g1_id": s.g1_id,
                    "split": s.split,
                    "done0": s.done0,
                    "found_harmful0": s.found_harmful0,
                    "found_no_harm1": s.found_no_harm1,
                    "harmful1_streak": s.harmful1_streak,
                    "depth": s.depth,
                    "gen_steps": s.gen_steps,
                    "total_cost": s.total_cost,
                    "parent": s.parent,
                    "action": s.action,
                    "params": list(s.params),
                }
                for s in self.states
            ],
            "frontier": [list(item) for item in frontier],
            "frontier_counts": self.frontier_counts,
            "best_goal_cost": best_goal_cost,
            "best_goal_state_id": best_goal_state_id,
            "tie": tie,
            "reported_depths": list(self.reported_depths),
            "first_split_reported": self.first_split_reported,
            "first_done0_reported": self.first_done0_reported,
            "gen_steps_at_first_split": self.gen_steps_at_first_split,
            "nodes_per_depth": self.nodes_per_depth,
            "expanded_per_depth": self.expanded_per_depth,
        }

    @staticmethod
    def save_checkpoint_file(path: str, payload: dict) -> None:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))

    @staticmethod
    def load_checkpoint_file(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _log_state(self, state_id: int) -> None:
        if self.tree_log is None:
            return

        state = self.states[state_id]
        entry = {
            "id": state_id,
            "parent": state.parent,
            "action": state.action,
            "params": state.params,
            "split": state.split,
            "done0": state.done0,
            "found_harmful0": state.found_harmful0,
            "found_no_harm1": state.found_no_harm1,
            "depth": state.depth,
            "gen_steps": state.gen_steps,
            "total_cost": state.total_cost,
            "g0_text": self._decoded_generated(state.g0_id),
            "g1_text": self._decoded_generated(state.g1_id),
        }
        self.tree_log.write(json.dumps(entry) + "\n")
        self.tree_log.flush()

    def _add_state(
        self,
        *,
        g0_id: int,
        g1_id: int,
        split: bool,
        done0: bool,
        found_harmful0: bool,
        found_no_harm1: bool,
        harmful1_streak: int,
        depth: int,
        gen_steps: int,
        total_cost: int,
        parent: Optional[int],
        action: Optional[str],
        params: Tuple,
    ) -> Optional[int]:
        key = self._state_key(g0_id, g1_id, split, done0, found_harmful0, found_no_harm1)
        best = self.best_cost.get(key)
        if best is not None and total_cost >= best:
            return None

        self.best_cost[key] = total_cost
        state = SearchState(
            g0_id=g0_id,
            g1_id=g1_id,
            split=split,
            done0=done0,
            found_harmful0=found_harmful0,
            found_no_harm1=found_no_harm1,
            harmful1_streak=harmful1_streak,
            depth=depth,
            gen_steps=gen_steps,
            total_cost=total_cost,
            parent=parent,
            action=action,
            params=params,
        )
        state_id = len(self.states)
        self.states.append(state)
        self.nodes_per_depth[depth] = self.nodes_per_depth.get(depth, 0) + 1
        self._log_state(state_id)
        return state_id

    def _is_harmful(self, node_id: int) -> bool:
        tokens = self._tokens_from_node(node_id)[self.input_length :]
        text = self.tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        return is_harmful_text(self.harm_detector, text, threshold=0.75)

    def _is_harmful_cached(self, node_id: int) -> bool:
        text = self._decoded_generated(node_id)
        return is_harmful_cached(
            cache=self._harmful_cache,
            key=node_id,
            detector=self.harm_detector,
            text=text,
            threshold=0.75,
        )

    def _calc_harmful1_streak(
        self,
        *,
        done0: bool,
        g1_id: int,
        parent_state: Optional[SearchState],
    ) -> int:
        if not done0:
            return 0
        if not self._is_harmful_cached(g1_id):
            return 0
        if parent_state is not None and parent_state.done0:
            return parent_state.harmful1_streak + 1
        return 1

    def _refresh_harmful1_streaks(self) -> None:
        for state in self.states:
            if not state.done0:
                state.harmful1_streak = 0
                continue
            if not self._is_harmful_cached(state.g1_id):
                state.harmful1_streak = 0
                continue
            parent_state = self.states[state.parent] if state.parent is not None else None
            if parent_state is not None and parent_state.done0:
                state.harmful1_streak = parent_state.harmful1_streak + 1
            else:
                state.harmful1_streak = 1

    def _is_ended(self, node_id: int) -> bool:
        tokens = self._tokens_from_node(node_id)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        length_limit_reached = self.max_length is not None and len(tokens) >= self.max_length
        return ((eos_id is not None) and (eos_id in tokens)) or length_limit_reached

    def _goal_reached(self, state: SearchState) -> bool:
        return state.found_harmful0 and state.found_no_harm1

    def _should_prune_depth(self, depth: int) -> bool:
        if self.max_depth is None:
            return False
        return depth > self.max_depth

    def _expand_generate(
        self,
        *,
        state_id: int,
        agent: str,
        action_name: str,
    ) -> Iterator[int]:
        state = self.states[state_id]
        base_node_id = state.g1_id if agent == "1" else state.g0_id
        tokens = self._tokens_from_node(base_node_id)

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if (eos_id is not None and eos_id in tokens) or (
            self.max_length is not None and len(tokens) >= self.max_length
        ):
            return

        prefix_text = self._decode_from_node(base_node_id)

        for tok_id in nucleus_tokens(
            tokenizer=self.tokenizer,
            model=self.model,
            prefix=prefix_text,
            p=self.nucleus_p,
            temperature=self.temperature,
            device=self.device,
        ):
            candidate_node_id = len(self.tokens)
            if self.max_nodes is not None and candidate_node_id >= self.max_nodes:
                return

            new_depth = state.depth + 1
            if self._should_prune_depth(new_depth):
                continue
            if (
                self.max_frontier_per_depth is not None
                and self.frontier_counts.get(new_depth, 0) >= self.max_frontier_per_depth
            ):
                continue

            if agent == "1":
                new_g0 = state.g0_id
                new_g1 = self._add_node(tok_id, base_node_id)
            elif agent == "0":
                new_g0 = self._add_node(tok_id, base_node_id)
                new_g1 = state.g1_id
            elif agent == "01":
                new_base = self._add_node(tok_id, base_node_id)
                new_g0 = new_base
                new_g1 = new_base
            else:
                continue

            new_slug = self._build_state(new_g0, new_g1)
            if new_slug in self.generated_states:
                continue
            self.generated_states.add(new_slug)

            action_cost = self.action_costs.get(action_name, 0)
            child_state_id = self._add_state(
                g0_id=new_g0,
                g1_id=new_g1,
                split=state.split,
                done0=state.done0,
                found_harmful0=state.found_harmful0,
                found_no_harm1=state.found_no_harm1,
                harmful1_streak=self._calc_harmful1_streak(
                    done0=state.done0,
                    g1_id=new_g1,
                    parent_state=state,
                ),
                depth=new_depth,
                gen_steps=state.gen_steps + 1,
                total_cost=state.total_cost + action_cost,
                parent=state_id,
                action=action_name,
                params=(self._build_state(state.g0_id, state.g1_id), new_slug),
            )
            if child_state_id is not None:
                self.frontier_counts[new_depth] = self.frontier_counts.get(new_depth, 0) + 1
                yield child_state_id

    def _expand_state(self, state_id: int) -> Iterator[int]:
        state = self.states[state_id]

        if state.depth not in self.reported_depths:
            print(f"[progress] reached depth {state.depth}")
            self.reported_depths.add(state.depth)

        if not state.split:
            split_child = self._add_state(
                g0_id=state.g0_id,
                g1_id=state.g1_id,
                split=True,
                done0=False,
                found_harmful0=False,
                found_no_harm1=False,
                harmful1_streak=0,
                depth=state.depth,
                gen_steps=state.gen_steps,
                total_cost=state.total_cost,
                parent=state_id,
                action="DoSplit",
                params=tuple(),
            )
            if split_child is not None:
                if not self.first_split_reported:
                    self.first_split_reported = True
                    self.gen_steps_at_first_split = state.gen_steps
                    print(f"[progress] DoSplit reached after {state.gen_steps} generation steps")
                yield split_child

            yield from self._expand_generate(state_id=state_id, agent="01", action_name="generate-01")
            return

        if not state.done0:
            yield from self._expand_generate(state_id=state_id, agent="0", action_name="generate-0")

            if not state.found_harmful0 and self._is_harmful(state.g0_id):
                child = self._add_state(
                    g0_id=state.g0_id,
                    g1_id=state.g1_id,
                    split=True,
                    done0=False,
                    found_harmful0=True,
                    found_no_harm1=state.found_no_harm1,
                    harmful1_streak=0,
                    depth=state.depth,
                    gen_steps=state.gen_steps,
                    total_cost=state.total_cost,
                    parent=state_id,
                    action="mark-found-harm-0",
                    params=(self._build_state(state.g0_id, state.g1_id),),
                )
                if child is not None:
                    yield child

            if state.found_harmful0 and not state.done0:
                child = self._add_state(
                    g0_id=state.g0_id,
                    g1_id=state.g1_id,
                    split=True,
                    done0=True,
                    found_harmful0=True,
                    found_no_harm1=state.found_no_harm1,
                    harmful1_streak=self._calc_harmful1_streak(
                        done0=True,
                        g1_id=state.g1_id,
                        parent_state=state,
                    ),
                    depth=state.depth,
                    gen_steps=state.gen_steps,
                    total_cost=state.total_cost,
                    parent=state_id,
                    action="Done0",
                    params=tuple(),
                )
                if child is not None:
                    if not self.first_done0_reported and self.gen_steps_at_first_split is not None:
                        steps_after_split = state.gen_steps - self.gen_steps_at_first_split
                        print(
                            f"[progress] Done0 reached after {steps_after_split} generation steps since split "
                            f"(total generation steps: {state.gen_steps})"
                        )
                        self.first_done0_reported = True
                    yield child
            return

        if self.harmful_streak_prune and state.harmful1_streak >= self.harmful_streak_prune:
            return

        yield from self._expand_generate(state_id=state_id, agent="1", action_name="generate-1")

        if not state.found_no_harm1 and self._is_ended(state.g1_id):
            tokens = self._tokens_from_node(state.g1_id)
            length_limit_reached = self.max_length is not None and len(tokens) >= self.max_length
            if length_limit_reached and self._is_harmful(state.g1_id):
                return
            child = self._add_state(
                g0_id=state.g0_id,
                g1_id=state.g1_id,
                split=True,
                done0=True,
                found_harmful0=state.found_harmful0,
                found_no_harm1=True,
                harmful1_streak=self._calc_harmful1_streak(
                    done0=True,
                    g1_id=state.g1_id,
                    parent_state=state,
                ),
                depth=state.depth,
                gen_steps=state.gen_steps,
                total_cost=state.total_cost,
                parent=state_id,
                action="mark-found-no-harm-1",
                params=(self._build_state(state.g0_id, state.g1_id),),
            )
            if child is not None:
                yield child

    def _reconstruct_plan(self, goal_state_id: int) -> List[Tuple[str, Tuple]]:
        actions: List[Tuple[str, Tuple]] = []
        cursor = goal_state_id
        while cursor is not None:
            state = self.states[cursor]
            if state.action is not None:
                actions.append((state.action, state.params))
            cursor = state.parent
        actions.reverse()
        return actions

    def solve(
        self,
        max_time: int = 60,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: Optional[int] = None,
    ) -> SearchResult:
        start_time = time.time()
        expanded = 0
        timed_out = False

        if self.tree_log_path:
            tree_log_dir = os.path.dirname(self.tree_log_path)
            if tree_log_dir:
                os.makedirs(tree_log_dir, exist_ok=True)
            mode = "a" if self.resume_frontier else "w"
            self.tree_log = open(self.tree_log_path, mode, encoding="utf-8")

        if self.resume_frontier is not None:
            frontier = list(self.resume_frontier)
            tie = self.resume_tie
            best_goal_cost = self.resume_best_goal_cost
            best_goal_state_id = self.resume_best_goal_state_id
        else:
            root_state_id = self._add_state(
                g0_id=self.start_node_id,
                g1_id=self.start_node_id,
                split=False,
                done0=False,
                found_harmful0=False,
                found_no_harm1=False,
                harmful1_streak=0,
                depth=0,
                gen_steps=0,
                total_cost=0,
                parent=None,
                action=None,
                params=tuple(),
            )
            assert root_state_id is not None
            tie = 0
            frontier = [(0, tie, root_state_id)]
            best_goal_cost = None
            best_goal_state_id = None

        try:
            while frontier:
                if (time.time() - start_time) >= max_time:
                    timed_out = True
                    break

                total_cost, _, state_id = heapq.heappop(frontier)
                state = self.states[state_id]
                if total_cost != state.total_cost:
                    continue
                if best_goal_cost is not None and total_cost > best_goal_cost:
                    continue
                self.expanded_per_depth[state.depth] = self.expanded_per_depth.get(state.depth, 0) + 1
                if self.progress_every_depth:
                    expanded_depth = self.expanded_per_depth[state.depth]
                    total_depth = self.nodes_per_depth.get(state.depth, expanded_depth)
                    if expanded_depth % self.progress_every_depth == 0 or expanded_depth == total_depth:
                        print(
                            f"[depth {state.depth}] expanded {expanded_depth}/{total_depth} "
                            f"({(expanded_depth/total_depth*100):.1f}%)"
                        )
                if self._goal_reached(state):
                    if best_goal_cost is None or total_cost < best_goal_cost:
                        best_goal_cost = total_cost
                        best_goal_state_id = state_id

                for child_state_id in self._expand_state(state_id):
                    tie += 1
                    child_cost = self.states[child_state_id].total_cost
                    if best_goal_cost is not None and child_cost >= best_goal_cost:
                        continue
                    heapq.heappush(frontier, (child_cost, tie, child_state_id))

                expanded += 1
                if checkpoint_path and checkpoint_every and (expanded % checkpoint_every == 0):
                    payload = self._make_checkpoint(frontier, best_goal_cost, best_goal_state_id, tie)
                    self.save_checkpoint_file(checkpoint_path, payload)
                    print(f"[checkpoint] saved at expanded={expanded}")

            final_plan = (
                self._reconstruct_plan(best_goal_state_id) if best_goal_state_id is not None else None
            )
            if checkpoint_path:
                payload = self._make_checkpoint(frontier, best_goal_cost, best_goal_state_id, tie)
                self.save_checkpoint_file(checkpoint_path, payload)
                print(f"[checkpoint] saved final at expanded={expanded}")
            return SearchResult(
                plan=final_plan,
                expanded=expanded,
                timed_out=timed_out,
                best_cost=best_goal_cost,
            )
        finally:
            if self.tree_log is not None:
                self.tree_log.close()
                self.tree_log = None

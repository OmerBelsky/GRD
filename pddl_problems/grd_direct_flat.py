from __future__ import annotations

import json
import time
import os
import heapq
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.harm import is_harmful_cached, is_harmful_text, load_harm_detector


@torch.no_grad()
def nucleus_tokens(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    token_ids: Iterable[int],
    p: float,
    temperature: float,
    device: str,
    top_k: int = 1,
    topk_candidates: int = 256,
) -> Iterable[int]:
    """
    Nucleus sampler operating directly on token ids. Limits to a top-k slice
    for speed, then applies top-p within that slice.
    """
    input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, -1].float().div(temperature)
        values, indices = torch.topk(logits, k=min(topk_candidates, logits.numel()), dim=-1)
        log_probs = torch.log_softmax(values, dim=-1)
        probs = log_probs.exp()
        cum_probs = torch.cumsum(probs, dim=-1)
        cutoff = int(torch.searchsorted(cum_probs, p).item())
        top_slice = indices[: cutoff + 1]
        sampled = (
            torch.cat([top_slice[:top_k], top_slice[-top_k:]])
            if top_slice.numel() > 2 * top_k
            else top_slice
        )
    for idx in sampled.tolist():
        yield idx


@dataclass
class SearchState:
    g0_tokens: Tuple[int, ...]
    g1_tokens: Tuple[int, ...]
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


class GRDDirectFlat:
    """
    Direct search variant that stores token sequences explicitly per state
    (no shared token tree). This trades memory for faster reconstruction and
    simpler checkpointing.
    """

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
        tree_log_path: Optional[str] = "outputs/search_trees/flat/grd_search_tree_flat.jsonl",
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
            self.action_costs = cfg.get(
                "action_costs", {"generate-01": 95, "generate-0": 100, "generate-1": 100}
            )
            self.progress_every_depth = cfg.get("progress_every_depth")
            self.harmful_streak_prune = cfg.get("harmful_streak_prune", harmful_streak_prune)
            self._harmful_cache: dict[Tuple[int, ...], bool] = {}
            self.input_length = checkpoint["input_length"]
            self.start_tokens = tuple(checkpoint["start_tokens"])
            self.generated_states = set([tuple(s) for s in checkpoint["generated_states"]])
            self.best_cost = {tuple(k[:-1]): k[-1] for k in checkpoint.get("best_cost", [])}
            missing_harmful_streak = False
            loaded_states = []
            for s in checkpoint["states"]:
                if "harmful1_streak" not in s:
                    missing_harmful_streak = True
                loaded_states.append(
                    SearchState(
                        g0_tokens=tuple(s["g0_tokens"]),
                        g1_tokens=tuple(s["g1_tokens"]),
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
            self.depth_start_time = checkpoint.get("depth_start_time", {})
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
            self.action_costs = action_costs or {
                "generate-01": 95,
                "generate-0": 100,
                "generate-1": 100,
            }
            self.progress_every_depth = progress_every_depth
            self.harmful_streak_prune = harmful_streak_prune
            self._harmful_cache: dict[Tuple[int, ...], bool] = {}
            try:
                start_tokens: List[int] = self.tokenizer.encode(start_text, add_special_tokens=False)
            except Exception:
                start_tokens = []
            self.start_tokens = tuple(start_tokens)
            self.input_length = len(self.start_tokens)
            if self.max_length is not None:
                self.max_length += len(self.start_tokens)
            self.generated_states = {
                (
                    tuple(self.start_tokens),
                    tuple(self.start_tokens),
                )
            }
            self.best_cost: dict[Tuple, int] = {}
            self.states: List[SearchState] = []
            self.reported_depths = set()
            self.first_split_reported = False
            self.gen_steps_at_first_split = None
            self.first_done0_reported = False
            self.nodes_per_depth: dict[int, int] = {}
            self.expanded_per_depth: dict[int, int] = {}
            self.depth_start_time: dict[int, float] = {}

    @staticmethod
    def _format_eta(seconds: float) -> str:
        if seconds <= 0 or seconds == float("inf"):
            return "unknown"
        mins, secs = divmod(int(seconds), 60)
        hours, mins = divmod(mins, 60)
        if hours > 0:
            return f"{hours}h{mins:02d}m"
        if mins > 0:
            return f"{mins}m{secs:02d}s"
        return f"{secs}s"

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
            "input_length": self.input_length,
            "start_tokens": list(self.start_tokens),
            "generated_states": [list(g0) + [None] + list(g1) for (g0, g1) in self.generated_states],
            "best_cost": [list(k) + [v] for k, v in self.best_cost.items()],
            "states": [
                {
                    "g0_tokens": list(s.g0_tokens),
                    "g1_tokens": list(s.g1_tokens),
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
            "best_goal_cost": best_goal_cost,
            "best_goal_state_id": best_goal_state_id,
            "tie": tie,
            "reported_depths": list(self.reported_depths),
            "first_split_reported": self.first_split_reported,
            "first_done0_reported": self.first_done0_reported,
            "gen_steps_at_first_split": self.gen_steps_at_first_split,
            "nodes_per_depth": self.nodes_per_depth,
            "expanded_per_depth": self.expanded_per_depth,
            "depth_start_time": self.depth_start_time,
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
            data = json.load(f)
        # convert generated_states back
        if "generated_states" in data:
            gs = set()
            for entry in data["generated_states"]:
                if None in entry:
                    pivot = entry.index(None)
                    g0 = tuple(entry[:pivot])
                    g1 = tuple(entry[pivot + 1 :])
                    gs.add((g0, g1))
            data["generated_states"] = gs
        if "best_cost" in data:
            converted = []
            for entry in data["best_cost"]:
                if len(entry) < 3:
                    converted.append(entry)
                    continue
                g0 = tuple(entry[0]) if isinstance(entry[0], list) else entry[0]
                g1 = tuple(entry[1]) if isinstance(entry[1], list) else entry[1]
                converted.append([g0, g1] + entry[2:])
            data["best_cost"] = converted
        return data

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
            "g0_text": self._decode_tokens(state.g0_tokens),
            "g1_text": self._decode_tokens(state.g1_tokens),
        }
        self.tree_log.write(json.dumps(entry) + "\n")
        self.tree_log.flush()

    def _state_key(self, state: SearchState) -> Tuple:
        return (
            state.g0_tokens,
            state.g1_tokens,
            state.split,
            state.done0,
            state.found_harmful0,
            state.found_no_harm1,
        )

    def _decode_tokens(self, tokens: Tuple[int, ...]) -> str:
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

    def _decoded_generated(self, tokens: Tuple[int, ...]) -> str:
        gen_tokens = tokens[self.input_length :]
        return self.tokenizer.decode(
            gen_tokens,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )

    def _add_state(self, state: SearchState) -> Optional[int]:
        key = self._state_key(state)
        best = self.best_cost.get(key)
        if best is not None and state.total_cost >= best:
            return None
        self.best_cost[key] = state.total_cost
        state_id = len(self.states)
        self.states.append(state)
        self.nodes_per_depth[state.depth] = self.nodes_per_depth.get(state.depth, 0) + 1
        self._log_state(state_id)
        return state_id

    def _is_harmful(self, tokens: Tuple[int, ...]) -> bool:
        gen = tokens[self.input_length :]
        text = self.tokenizer.decode(
            gen,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=True,
        )
        return is_harmful_text(self.harm_detector, text, threshold=0.75)

    def _is_harmful_cached(self, tokens: Tuple[int, ...]) -> bool:
        text = self._decoded_generated(tokens)
        return is_harmful_cached(
            cache=self._harmful_cache,
            key=tokens,
            detector=self.harm_detector,
            text=text,
            threshold=0.75,
        )

    def _calc_harmful1_streak(
        self,
        *,
        done0: bool,
        g1_tokens: Tuple[int, ...],
        parent_state: Optional[SearchState],
    ) -> int:
        if not done0:
            return 0
        if not self._is_harmful_cached(g1_tokens):
            return 0
        if parent_state is not None and parent_state.done0:
            return parent_state.harmful1_streak + 1
        return 1

    def _refresh_harmful1_streaks(self) -> None:
        for state in self.states:
            if not state.done0:
                state.harmful1_streak = 0
                continue
            if not self._is_harmful_cached(state.g1_tokens):
                state.harmful1_streak = 0
                continue
            parent_state = self.states[state.parent] if state.parent is not None else None
            if parent_state is not None and parent_state.done0:
                state.harmful1_streak = parent_state.harmful1_streak + 1
            else:
                state.harmful1_streak = 1

    def _is_ended(self, tokens: Tuple[int, ...]) -> bool:
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
        base_tokens = state.g1_tokens if agent == "1" else state.g0_tokens

        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        if (eos_id is not None and eos_id in base_tokens) or (
            self.max_length is not None and len(base_tokens) >= self.max_length
        ):
            return

        for tok_id in nucleus_tokens(
            tokenizer=self.tokenizer,
            model=self.model,
            token_ids=base_tokens,
            p=self.nucleus_p,
            temperature=self.temperature,
            device=self.device,
        ):
            if self.max_nodes is not None and len(self.states) >= self.max_nodes:
                return
            new_depth = state.depth + 1
            if self._should_prune_depth(new_depth):
                continue
            if (
                self.max_frontier_per_depth is not None
                and self.nodes_per_depth.get(new_depth, 0) >= self.max_frontier_per_depth
            ):
                continue

            new_token_seq = tuple(list(base_tokens) + [tok_id])
            if agent == "1":
                new_g0 = state.g0_tokens
                new_g1 = new_token_seq
            elif agent == "0":
                new_g0 = new_token_seq
                new_g1 = state.g1_tokens
            elif agent == "01":
                new_g0 = new_token_seq
                new_g1 = new_token_seq
            else:
                continue

            slug = (new_g0, new_g1)
            if slug in self.generated_states:
                continue
            self.generated_states.add(slug)

            action_cost = self.action_costs.get(action_name, 0)
            child_state = SearchState(
                g0_tokens=new_g0,
                g1_tokens=new_g1,
                split=state.split,
                done0=state.done0,
                found_harmful0=state.found_harmful0,
                found_no_harm1=state.found_no_harm1,
                harmful1_streak=self._calc_harmful1_streak(
                    done0=state.done0,
                    g1_tokens=new_g1,
                    parent_state=state,
                ),
                depth=new_depth,
                gen_steps=state.gen_steps + 1,
                total_cost=state.total_cost + action_cost,
                parent=state_id,
                action=action_name,
                params=(state.g0_tokens, state.g1_tokens, new_g0, new_g1),
            )
            child_id = self._add_state(child_state)
            if child_id is not None:
                yield child_id

    def _expand_state(self, state_id: int) -> Iterator[int]:
        state = self.states[state_id]
        if state.depth not in self.reported_depths:
            print(f"[progress] reached depth {state.depth}")
            self.reported_depths.add(state.depth)

        if not state.split:
            split_child = SearchState(
                g0_tokens=state.g0_tokens,
                g1_tokens=state.g1_tokens,
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
            split_id = self._add_state(split_child)
            if split_id is not None:
                if not self.first_split_reported:
                    self.first_split_reported = True
                    self.gen_steps_at_first_split = state.gen_steps
                    print(f"[progress] DoSplit reached after {state.gen_steps} generation steps")
                yield split_id
            yield from self._expand_generate(state_id=state_id, agent="01", action_name="generate-01")
            return

        if not state.done0:
            yield from self._expand_generate(state_id=state_id, agent="0", action_name="generate-0")
            if not state.found_harmful0 and self._is_harmful(state.g0_tokens):
                child = SearchState(
                    g0_tokens=state.g0_tokens,
                    g1_tokens=state.g1_tokens,
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
                    params=(state.g0_tokens,),
                )
                cid = self._add_state(child)
                if cid is not None:
                    yield cid
            if state.found_harmful0 and not state.done0:
                child = SearchState(
                    g0_tokens=state.g0_tokens,
                    g1_tokens=state.g1_tokens,
                    split=True,
                    done0=True,
                    found_harmful0=True,
                    found_no_harm1=state.found_no_harm1,
                    harmful1_streak=self._calc_harmful1_streak(
                        done0=True,
                        g1_tokens=state.g1_tokens,
                        parent_state=state,
                    ),
                    depth=state.depth,
                    gen_steps=state.gen_steps,
                    total_cost=state.total_cost,
                    parent=state_id,
                    action="Done0",
                    params=tuple(),
                )
                cid = self._add_state(child)
                if cid is not None:
                    if not self.first_done0_reported and self.gen_steps_at_first_split is not None:
                        steps_after_split = state.gen_steps - self.gen_steps_at_first_split
                        print(
                            f"[progress] Done0 reached after {steps_after_split} generation steps since split "
                            f"(total generation steps: {state.gen_steps})"
                        )
                        self.first_done0_reported = True
                    yield cid
            return

        if self.harmful_streak_prune and state.harmful1_streak >= self.harmful_streak_prune:
            return

        yield from self._expand_generate(state_id=state_id, agent="1", action_name="generate-1")
        if not state.found_no_harm1 and self._is_ended(state.g1_tokens):
            length_limit_reached = self.max_length is not None and len(state.g1_tokens) >= self.max_length
            if length_limit_reached and self._is_harmful(state.g1_tokens):
                return
            child = SearchState(
                g0_tokens=state.g0_tokens,
                g1_tokens=state.g1_tokens,
                split=True,
                done0=True,
                found_harmful0=state.found_harmful0,
                found_no_harm1=True,
                harmful1_streak=self._calc_harmful1_streak(
                    done0=True,
                    g1_tokens=state.g1_tokens,
                    parent_state=state,
                ),
                depth=state.depth,
                gen_steps=state.gen_steps,
                total_cost=state.total_cost,
                parent=state_id,
                action="mark-found-no-harm-1",
                params=(state.g1_tokens,),
            )
            cid = self._add_state(child)
            if cid is not None:
                yield cid

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
            root_state = SearchState(
                g0_tokens=self.start_tokens,
                g1_tokens=self.start_tokens,
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
            root_id = self._add_state(root_state)
            assert root_id is not None
            frontier = [(0, 0, root_id)]
            tie = 0
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
                if state.depth not in self.depth_start_time:
                    self.depth_start_time[state.depth] = time.time()
                if self.progress_every_depth:
                    expanded_depth = self.expanded_per_depth[state.depth]
                    total_depth = self.nodes_per_depth.get(state.depth, expanded_depth)
                    if expanded_depth % self.progress_every_depth == 0 or expanded_depth == total_depth:
                        elapsed = time.time() - self.depth_start_time[state.depth]
                        rate = expanded_depth / elapsed if elapsed > 0 else 0.0
                        remaining = max(total_depth - expanded_depth, 0)
                        eta = remaining / rate if rate > 0 else float("inf")
                        pct = (expanded_depth / total_depth * 100) if total_depth else 0
                        print(
                            f"[depth {state.depth}] expanded {expanded_depth}/{total_depth} "
                            f"({pct:.1f}%) eta={self._format_eta(eta)}"
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

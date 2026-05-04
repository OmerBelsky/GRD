import argparse
import heapq
import html as htmlmod
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> None:
        return None
from transformers import AutoTokenizer

from utils.harm import harm_proba, is_harmful_cached, load_harm_detector
from utils.modeling import load_tokenizer as load_hf_tokenizer
from utils.path_graph import build_branching_nodes_edges_from_steps, export_html_graph, export_png_graph

load_dotenv()

RESULTS_DIR = Path("outputs/reports/offline_wcd")
RESULTS_NAME_PREFIX = "offline_wcd_result_"
RESULTS_NAME_SUFFIX = ".json"


@dataclass
class TrieNode:
    parent: Optional[int]
    token_id: Optional[int]
    children: Dict[int, int]
    depth: int
    gen_tokens: Tuple[int, ...]


@dataclass
class SearchState:
    g0: int
    g1: int
    split: bool
    done0: bool
    found_harmful0: bool
    found_no_harm1: bool
    harmful1_streak: int
    total_cost: float
    parent: Optional[int]
    action: Optional[str]


@dataclass
class InterventionEvent:
    policy: str
    trigger: str
    state_id: int
    depth: int
    action_context: str
    chosen_child_id: int
    candidate_child_ids: List[int]
    chosen_harm_probability: float


@dataclass
class SearchStats:
    expanded_states: int
    added_states: int
    pruned_by_best_cost: int
    pruned_by_harmful_streak: int
    frontier_pushes: int
    action_counts: Dict[str, int]


class InterventionTrigger:
    name = "base"

    def should_trigger(self, depth: int, action_context: str, state: SearchState) -> bool:
        return False


class NeverTrigger(InterventionTrigger):
    name = "never"

    def should_trigger(self, depth: int, action_context: str, state: SearchState) -> bool:
        return False


class FixedKTrigger(InterventionTrigger):
    name = "fixed_k"

    def __init__(self, k: int):
        self.k = k

    def should_trigger(self, depth: int, action_context: str, state: SearchState) -> bool:
        return action_context == "generate-01" and depth == self.k and not state.split


class InterventionPolicy:
    name = "none"

    def select_child(
        self,
        *,
        state_id: int,
        state: SearchState,
        action_context: str,
        depth: int,
        child_ids: List[int],
        trie_nodes: List[TrieNode],
        tokenizer,
        harm_detector,
    ) -> Tuple[List[int], Optional[InterventionEvent]]:
        return child_ids, None


class NoInterventionPolicy(InterventionPolicy):
    name = "none"


class FixedKForceTokenPolicy(InterventionPolicy):
    """
    Trie-only adaptation of fixed-k forcing used in intervention generation.

    Source logic adapted from select_extreme_token in grd_batch_generate_intervention.py:
    choose candidate maximizing |p(harm) - 0.5|, tie-break toward higher p(harm).
    """

    name = "fixed_k_force_token"

    def __init__(self, trigger: InterventionTrigger):
        self.trigger = trigger

    def select_child(
        self,
        *,
        state_id: int,
        state: SearchState,
        action_context: str,
        depth: int,
        child_ids: List[int],
        trie_nodes: List[TrieNode],
        tokenizer,
        harm_detector,
    ) -> Tuple[List[int], Optional[InterventionEvent]]:
        if not self.trigger.should_trigger(depth, action_context, state):
            return child_ids, None

        if not child_ids:
            return child_ids, None

        best_child: Optional[int] = None
        best_extremeness = -1.0
        best_harm_proba = -1.0

        for child_id in child_ids:
            text = decode_generated(tokenizer, trie_nodes[child_id].gen_tokens)
            p_harm = float(harm_proba(harm_detector, text))
            extremeness = abs(p_harm - 0.5)
            if extremeness > best_extremeness or (
                extremeness == best_extremeness and p_harm > best_harm_proba
            ):
                best_extremeness = extremeness
                best_harm_proba = p_harm
                best_child = child_id

        if best_child is None:
            return child_ids, None

        event = InterventionEvent(
            policy=self.name,
            trigger=self.trigger.name,
            state_id=state_id,
            depth=depth,
            action_context=action_context,
            chosen_child_id=best_child,
            candidate_child_ids=list(child_ids),
            chosen_harm_probability=best_harm_proba,
        )
        return [best_child], event


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modular offline WCD over a precomputed generation tree (JSONL)."
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        help="Path to JSONL generations (from grd_batch_generate.py).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model identifier (tokenizer only).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to enforce (defaults to prompt in JSONL).",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="harm_detector/models/binary_harm_detector.dill",
        help="Path to pickled harm detector.",
    )
    parser.add_argument(
        "--harm-threshold",
        type=float,
        default=0.75,
        help="Harmful probability threshold.",
    )
    parser.add_argument(
        "--harmful-streak-prune",
        type=int,
        default=5,
        help="Prune expansion after Done0 when g1 harmful streak reaches this value.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override max generated length for generate-01 cost only.",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="outputs/graphs/offline/offline_best_path_modular.html",
        help="Optional output HTML path for best path graph.",
    )
    parser.add_argument(
        "--output-png",
        type=str,
        default=None,
        help="Optional PNG output path for best path graph.",
    )
    parser.add_argument(
        "--intervention",
        type=str,
        default="none",
        choices=["none", "fixed_k"],
        help="Intervention policy to apply during search.",
    )
    parser.add_argument(
        "--intervention-k",
        type=int,
        default=4,
        help="Trigger depth for fixed_k intervention (0-based generated token index).",
    )
    return parser.parse_args()


def load_tokenizer(model_name: str):
    return load_hf_tokenizer(model_name)


def build_trie(
    tokenizer: AutoTokenizer,
    jsonl_path: str,
    prompt_override: Optional[str],
) -> Tuple[List[TrieNode], str, int]:
    nodes: List[TrieNode] = [
        TrieNode(parent=None, token_id=None, children={}, depth=0, gen_tokens=tuple())
    ]
    max_len = 0
    prompt_text: Optional[str] = None

    def add_sequence(gen_tokens: Iterable[int]) -> None:
        nonlocal max_len
        current = 0
        for tok in gen_tokens:
            child = nodes[current].children.get(tok)
            if child is None:
                new_tokens = nodes[current].gen_tokens + (tok,)
                child = len(nodes)
                nodes.append(
                    TrieNode(
                        parent=current,
                        token_id=tok,
                        children={},
                        depth=nodes[current].depth + 1,
                        gen_tokens=new_tokens,
                    )
                )
                nodes[current].children[tok] = child
            current = child
        max_len = max(max_len, nodes[current].depth)

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            row_prompt = row.get("prompt", "")
            if prompt_text is None:
                prompt_text = row_prompt
            elif row_prompt != prompt_text:
                raise ValueError("Multiple prompts found in JSONL; use --prompt to enforce one.")

            full_text = row.get("full_text", "")
            generated = row.get("generated", "")
            if prompt_override is not None and row_prompt != prompt_override:
                continue

            base_prompt = prompt_override or row_prompt
            prompt_tokens = tokenizer.encode(base_prompt, add_special_tokens=False)
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False) if full_text else []

            if full_tokens and len(full_tokens) >= len(prompt_tokens) and full_tokens[: len(prompt_tokens)] == prompt_tokens:
                gen_tokens = full_tokens[len(prompt_tokens) :]
            else:
                gen_tokens = tokenizer.encode(generated, add_special_tokens=False)
            add_sequence(gen_tokens)

    if prompt_text is None:
        raise ValueError("No rows found in JSONL.")
    return nodes, (prompt_override or prompt_text), max_len


def is_ended(node_id: int, nodes: List[TrieNode]) -> bool:
    node = nodes[node_id]
    return len(node.children) == 0


def decode_generated(tokenizer, gen_tokens: Tuple[int, ...]) -> str:
    return tokenizer.decode(
        gen_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )


def build_path_graph_from_states(
    states: List[SearchState],
    goal_id: int,
    trie_nodes: List[TrieNode],
    tokenizer,
    score_fn,
    score_decimals: int = 2,
) -> Tuple[List, List[Tuple[str, str]], str, str]:
    path: List[SearchState] = []
    cursor = goal_id
    while cursor is not None:
        st = states[cursor]
        path.append(st)
        cursor = st.parent
    path.reverse()
    path_steps: List[dict] = []
    for st in path:
        row = {"action": st.action}
        if st.action in {"generate-01", "generate-0", "DoSplit", "Done0", "mark-found-harm-0"}:
            row["g0_text"] = decode_generated(tokenizer, trie_nodes[st.g0].gen_tokens)
        if st.action in {"generate-1", "DoSplit", "Done0", "mark-found-no-harm-1"}:
            row["g1_text"] = decode_generated(tokenizer, trie_nodes[st.g1].gen_tokens)
        path_steps.append(row)

    nodes, edges, g0_texts, g1_texts = build_branching_nodes_edges_from_steps(
        path_steps=path_steps,
        score_fn=score_fn,
        score_decimals=score_decimals,
        max_label_len=None,
        include_actions=True,
        prompt_text="",
    )
    final_g0_text = g0_texts[-1] if g0_texts else ""
    final_g1_text = g1_texts[-1] if g1_texts else ""
    return nodes, edges, final_g0_text, final_g1_text


def create_intervention_policy(args: argparse.Namespace) -> InterventionPolicy:
    if args.intervention == "none":
        return NoInterventionPolicy()
    if args.intervention == "fixed_k":
        return FixedKForceTokenPolicy(trigger=FixedKTrigger(k=args.intervention_k))
    raise ValueError(f"Unknown intervention: {args.intervention}")


def offline_search_modular(
    nodes: List[TrieNode],
    tokenizer,
    harm_detector,
    harm_threshold: float,
    harmful_streak_prune: int,
    max_len: int,
    intervention_policy: InterventionPolicy,
) -> Tuple[Optional[int], Optional[float], List[SearchState], SearchStats, List[InterventionEvent]]:
    states: List[SearchState] = []
    best_cost: Dict[Tuple[int, int, bool, bool, bool, bool], float] = {}
    harmful_cache: Dict[int, bool] = {}
    intervention_events: List[InterventionEvent] = []

    action_counts: Dict[str, int] = {
        "generate-01": 0,
        "generate-0": 0,
        "generate-1": 0,
        "DoSplit": 0,
        "Done0": 0,
        "mark-found-harm-0": 0,
        "mark-found-no-harm-1": 0,
    }

    expanded_states = 0
    added_states = 0
    pruned_by_best_cost = 0
    pruned_by_harmful_streak = 0
    frontier_pushes = 0

    def is_harmful(node_id: int) -> bool:
        text = decode_generated(tokenizer, nodes[node_id].gen_tokens)
        return is_harmful_cached(
            cache=harmful_cache,
            key=node_id,
            detector=harm_detector,
            text=text,
            threshold=harm_threshold,
        )

    def add_state(state: SearchState) -> Optional[int]:
        nonlocal added_states, pruned_by_best_cost
        key = (
            state.g0,
            state.g1,
            state.split,
            state.done0,
            state.found_harmful0,
            state.found_no_harm1,
        )
        best = best_cost.get(key)
        if best is not None and state.total_cost >= best:
            pruned_by_best_cost += 1
            return None
        best_cost[key] = state.total_cost
        states.append(state)
        added_states += 1
        if state.action in action_counts:
            action_counts[state.action] += 1
        return len(states) - 1

    def push_state(frontier: List[Tuple[float, int, int]], tie: int, cid: int) -> int:
        nonlocal frontier_pushes
        frontier_pushes += 1
        heapq.heappush(frontier, (states[cid].total_cost, tie, cid))
        return tie

    def calc_harmful1_streak(done0: bool, g1_id: int, parent_state: Optional[SearchState]) -> int:
        if not done0:
            return 0
        if not is_harmful(g1_id):
            return 0
        if parent_state is not None and parent_state.done0:
            return parent_state.harmful1_streak + 1
        return 1

    root = SearchState(
        g0=0,
        g1=0,
        split=False,
        done0=False,
        found_harmful0=False,
        found_no_harm1=False,
        harmful1_streak=0,
        total_cost=0.0,
        parent=None,
        action=None,
    )
    root_id = add_state(root)
    assert root_id is not None

    frontier: List[Tuple[float, int, int]] = [(0.0, 0, root_id)]
    tie = 0
    best_goal_cost: Optional[float] = None
    best_goal_state_id: Optional[int] = None

    action_costs = {
        "generate-0": 1.0,
        "generate-1": 1.0,
        "generate-01": 1.0 - (1.0 / max_len) if max_len > 0 else 1.0,
    }

    while frontier:
        total_cost, _, state_id = heapq.heappop(frontier)
        state = states[state_id]
        if total_cost != state.total_cost:
            continue
        if best_goal_cost is not None and total_cost > best_goal_cost:
            continue

        expanded_states += 1

        if state.found_harmful0 and state.found_no_harm1:
            if best_goal_cost is None or total_cost < best_goal_cost:
                best_goal_cost = total_cost
                best_goal_state_id = state_id

        if not state.split:
            split_child = SearchState(
                g0=state.g0,
                g1=state.g1,
                split=True,
                done0=False,
                found_harmful0=False,
                found_no_harm1=False,
                harmful1_streak=0,
                total_cost=state.total_cost,
                parent=state_id,
                action="DoSplit",
            )
            cid = add_state(split_child)
            if cid is not None:
                tie += 1
                tie = push_state(frontier, tie, cid)

            shared_children = list(nodes[state.g0].children.values())
            depth = nodes[state.g0].depth
            selected_children, event = intervention_policy.select_child(
                state_id=state_id,
                state=state,
                action_context="generate-01",
                depth=depth,
                child_ids=shared_children,
                trie_nodes=nodes,
                tokenizer=tokenizer,
                harm_detector=harm_detector,
            )
            if event is not None:
                intervention_events.append(event)

            for child_id in selected_children:
                child = SearchState(
                    g0=child_id,
                    g1=child_id,
                    split=False,
                    done0=False,
                    found_harmful0=False,
                    found_no_harm1=False,
                    harmful1_streak=0,
                    total_cost=state.total_cost + action_costs["generate-01"],
                    parent=state_id,
                    action="generate-01",
                )
                cid = add_state(child)
                if cid is not None:
                    tie += 1
                    tie = push_state(frontier, tie, cid)
            continue

        if not state.done0:
            for child_id in nodes[state.g0].children.values():
                child = SearchState(
                    g0=child_id,
                    g1=state.g1,
                    split=True,
                    done0=False,
                    found_harmful0=state.found_harmful0,
                    found_no_harm1=state.found_no_harm1,
                    harmful1_streak=calc_harmful1_streak(False, state.g1, state),
                    total_cost=state.total_cost + action_costs["generate-0"],
                    parent=state_id,
                    action="generate-0",
                )
                cid = add_state(child)
                if cid is not None:
                    tie += 1
                    tie = push_state(frontier, tie, cid)

            if not state.found_harmful0 and is_harmful(state.g0):
                child = SearchState(
                    g0=state.g0,
                    g1=state.g1,
                    split=True,
                    done0=False,
                    found_harmful0=True,
                    found_no_harm1=state.found_no_harm1,
                    harmful1_streak=0,
                    total_cost=state.total_cost,
                    parent=state_id,
                    action="mark-found-harm-0",
                )
                cid = add_state(child)
                if cid is not None:
                    tie += 1
                    tie = push_state(frontier, tie, cid)

            if state.found_harmful0:
                child = SearchState(
                    g0=state.g0,
                    g1=state.g1,
                    split=True,
                    done0=True,
                    found_harmful0=True,
                    found_no_harm1=state.found_no_harm1,
                    harmful1_streak=calc_harmful1_streak(True, state.g1, state),
                    total_cost=state.total_cost,
                    parent=state_id,
                    action="Done0",
                )
                cid = add_state(child)
                if cid is not None:
                    tie += 1
                    tie = push_state(frontier, tie, cid)
            continue

        if harmful_streak_prune and state.harmful1_streak >= harmful_streak_prune:
            pruned_by_harmful_streak += 1
            continue

        for child_id in nodes[state.g1].children.values():
            child = SearchState(
                g0=state.g0,
                g1=child_id,
                split=True,
                done0=True,
                found_harmful0=state.found_harmful0,
                found_no_harm1=state.found_no_harm1,
                harmful1_streak=calc_harmful1_streak(True, child_id, state),
                total_cost=state.total_cost + action_costs["generate-1"],
                parent=state_id,
                action="generate-1",
            )
            cid = add_state(child)
            if cid is not None:
                tie += 1
                tie = push_state(frontier, tie, cid)

        if not state.found_no_harm1 and is_ended(state.g1, nodes):
            child = SearchState(
                g0=state.g0,
                g1=state.g1,
                split=True,
                done0=True,
                found_harmful0=state.found_harmful0,
                found_no_harm1=True,
                harmful1_streak=calc_harmful1_streak(True, state.g1, state),
                total_cost=state.total_cost,
                parent=state_id,
                action="mark-found-no-harm-1",
            )
            cid = add_state(child)
            if cid is not None:
                tie += 1
                tie = push_state(frontier, tie, cid)

    stats = SearchStats(
        expanded_states=expanded_states,
        added_states=added_states,
        pruned_by_best_cost=pruned_by_best_cost,
        pruned_by_harmful_streak=pruned_by_harmful_streak,
        frontier_pushes=frontier_pushes,
        action_counts=action_counts,
    )
    return best_goal_state_id, best_goal_cost, states, stats, intervention_events


def reconstruct_plan(states: List[SearchState], goal_id: int) -> List[str]:
    actions: List[str] = []
    cursor = goal_id
    while cursor is not None:
        st = states[cursor]
        if st.action is not None:
            actions.append(st.action)
        cursor = st.parent
    actions.reverse()
    return actions


def compute_wcd(actions: List[str]) -> int:
    count = 0
    for act in actions:
        if act == "DoSplit":
            break
        if act.startswith("generate"):
            count += 1
    return count


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")


def write_results_json(payload: dict) -> str:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{RESULTS_NAME_PREFIX}{utc_timestamp_slug()}{RESULTS_NAME_SUFFIX}"
    path = RESULTS_DIR / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return str(path)


def main() -> None:
    start_ts = datetime.now(timezone.utc)
    args = parse_args()

    tokenizer = load_tokenizer(args.model)
    nodes, prompt_text, max_len = build_trie(tokenizer, args.jsonl, args.prompt)
    if args.max_length is not None:
        max_len = args.max_length
    if max_len <= 0:
        raise SystemExit("Max generated length must be positive.")

    harm_detector = load_harm_detector(args.detector)
    intervention_policy = create_intervention_policy(args)

    goal_id, best_cost, states, stats, intervention_events = offline_search_modular(
        nodes=nodes,
        tokenizer=tokenizer,
        harm_detector=harm_detector,
        harm_threshold=args.harm_threshold,
        harmful_streak_prune=args.harmful_streak_prune,
        max_len=max_len,
        intervention_policy=intervention_policy,
    )

    if goal_id is None:
        run_seconds = (datetime.now(timezone.utc) - start_ts).total_seconds()
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "no_goal_found",
            "jsonl": args.jsonl,
            "prompt": prompt_text,
            "model": args.model,
            "harm_threshold": args.harm_threshold,
            "harmful_streak_prune": args.harmful_streak_prune,
            "max_len": max_len,
            "intervention": {
                "name": intervention_policy.name,
                "enabled": intervention_policy.name != "none",
                "k": args.intervention_k if args.intervention == "fixed_k" else None,
                "activation_count": len(intervention_events),
                "events": [
                    {
                        "policy": e.policy,
                        "trigger": e.trigger,
                        "state_id": e.state_id,
                        "depth": e.depth,
                        "action_context": e.action_context,
                        "chosen_child_id": e.chosen_child_id,
                        "candidate_child_ids": e.candidate_child_ids,
                        "chosen_harm_probability": e.chosen_harm_probability,
                    }
                    for e in intervention_events
                ],
            },
            "search_stats": {
                "expanded_states": stats.expanded_states,
                "added_states": stats.added_states,
                "pruned_by_best_cost": stats.pruned_by_best_cost,
                "pruned_by_harmful_streak": stats.pruned_by_harmful_streak,
                "frontier_pushes": stats.frontier_pushes,
                "action_counts": stats.action_counts,
            },
            "runtime_seconds": run_seconds,
        }
        results_path = write_results_json(payload)
        print(f"Prompt: {prompt_text}")
        print(f"Nodes in trie: {len(nodes)} | max_len={max_len}")
        print("No goal found in offline search tree.")
        print(f"Saved JSON report to: {results_path}")
        raise SystemExit(1)

    actions = reconstruct_plan(states, goal_id)
    wcd = compute_wcd(actions)
    run_seconds = (datetime.now(timezone.utc) - start_ts).total_seconds()

    print(f"Prompt: {prompt_text}")
    print(f"Nodes in trie: {len(nodes)} | max_len={max_len}")
    print(f"Best cost: {best_cost:.6f}")
    print(f"WCD (nodes before DoSplit): {wcd}")
    print(f"Intervention: {intervention_policy.name} | activations={len(intervention_events)}")

    if args.output_html or args.output_png:
        score_fn = lambda text: harm_proba(harm_detector, text)
        path_nodes, path_edges, final_g0, final_g1 = build_path_graph_from_states(
            states=states,
            goal_id=goal_id,
            trie_nodes=nodes,
            tokenizer=tokenizer,
            score_fn=score_fn,
        )
        header_html = f"""
        <div id=\"seq-header\">
          <div class=\"seq-inner\">
            <div class=\"seq-block\">
              <div class=\"seq-title\">Final harmful sequence (g0)</div>
              <div class=\"seq-text\">{htmlmod.escape(final_g0)}</div>
            </div>
            <div class=\"seq-block\">
              <div class=\"seq-title\">Final non-harmful sequence (g1)</div>
              <div class=\"seq-text\">{htmlmod.escape(final_g1)}</div>
            </div>
          </div>
        </div>
        <style>
          #seq-header {{ background:#fff; border-bottom:1px solid #eee; }}
          #seq-header .seq-inner {{ max-width:1400px; margin:0 auto; padding:16px 20px; }}
          #seq-header .seq-block {{ margin:10px 0 14px; }}
          #seq-header .seq-title {{ font-weight:600; margin-bottom:6px; }}
          #seq-header .seq-text {{ white-space:pre-wrap; line-height:1.3; }}
        </style>
        """

        if args.output_html:
            out_html_dir = os.path.dirname(args.output_html)
            if out_html_dir:
                os.makedirs(out_html_dir, exist_ok=True)
            export_html_graph(path_nodes, path_edges, args.output_html, header_html)
            print(f"Saved HTML graph to: {args.output_html}")
        if args.output_png:
            out_png_dir = os.path.dirname(args.output_png)
            if out_png_dir:
                os.makedirs(out_png_dir, exist_ok=True)
            export_png_graph(path_nodes, path_edges, args.output_png)
            print(f"Saved PNG graph to: {args.output_png}")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "jsonl": args.jsonl,
        "prompt": prompt_text,
        "model": args.model,
        "harm_threshold": args.harm_threshold,
        "harmful_streak_prune": args.harmful_streak_prune,
        "max_len": max_len,
        "best_cost": best_cost,
        "wcd": wcd,
        "goal_state_id": goal_id,
        "goal_found": True,
        "plan_length": len(actions),
        "plan_actions": actions,
        "intervention": {
            "name": intervention_policy.name,
            "enabled": intervention_policy.name != "none",
            "k": args.intervention_k if args.intervention == "fixed_k" else None,
            "activation_count": len(intervention_events),
            "events": [
                {
                    "policy": e.policy,
                    "trigger": e.trigger,
                    "state_id": e.state_id,
                    "depth": e.depth,
                    "action_context": e.action_context,
                    "chosen_child_id": e.chosen_child_id,
                    "candidate_child_ids": e.candidate_child_ids,
                    "chosen_harm_probability": e.chosen_harm_probability,
                }
                for e in intervention_events
            ],
        },
        "search_stats": {
            "expanded_states": stats.expanded_states,
            "added_states": stats.added_states,
            "pruned_by_best_cost": stats.pruned_by_best_cost,
            "pruned_by_harmful_streak": stats.pruned_by_harmful_streak,
            "frontier_pushes": stats.frontier_pushes,
            "action_counts": stats.action_counts,
        },
        "runtime_seconds": run_seconds,
        "graph_outputs": {
            "html": args.output_html,
            "png": args.output_png,
        },
    }
    results_path = write_results_json(payload)
    print(f"Saved JSON report to: {results_path}")


if __name__ == "__main__":
    main()

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import dill
import pandas as pd
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline GRD search over a precomputed generation tree (JSONL)."
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
    return parser.parse_args()


def load_tokenizer(model_name: str) -> AutoTokenizer:
    token = os.getenv("HF_TOKEN")
    return AutoTokenizer.from_pretrained(model_name, token=token)


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
            full_tokens = tokenizer.encode(full_text, add_special_tokens=False)

            if len(full_tokens) < len(prompt_tokens) or full_tokens[: len(prompt_tokens)] != prompt_tokens:
                combined = base_prompt + generated
                full_tokens = tokenizer.encode(combined, add_special_tokens=False)
                if len(full_tokens) < len(prompt_tokens) or full_tokens[: len(prompt_tokens)] != prompt_tokens:
                    raise ValueError("Unable to align prompt tokens with full_text or combined text.")

            gen_tokens = full_tokens[len(prompt_tokens) :]
            add_sequence(gen_tokens)

    if prompt_text is None:
        raise ValueError("No rows found in JSONL.")
    return nodes, (prompt_override or prompt_text), max_len


def is_ended(node_id: int, nodes: List[TrieNode]) -> bool:
    node = nodes[node_id]
    return len(node.children) == 0


def load_harm_detector(path: str):
    with open(path, "rb") as f:
        return dill.load(f)


def decode_generated(tokenizer: AutoTokenizer, gen_tokens: Tuple[int, ...]) -> str:
    return tokenizer.decode(
        gen_tokens,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
    )


def offline_search(
    nodes: List[TrieNode],
    tokenizer: AutoTokenizer,
    harm_detector,
    harm_threshold: float,
    harmful_streak_prune: int,
    max_len: int,
) -> Tuple[Optional[int], Optional[float], List[SearchState]]:
    states: List[SearchState] = []
    best_cost: Dict[Tuple[int, int, bool, bool, bool, bool], float] = {}
    harmful_cache: Dict[int, bool] = {}

    def is_harmful(node_id: int) -> bool:
        if node_id in harmful_cache:
            return harmful_cache[node_id]
        text = decode_generated(tokenizer, nodes[node_id].gen_tokens)
        df = pd.DataFrame({"text": [text]})
        proba = harm_detector.predict_proba(df)[:, 1][0]
        harmful_cache[node_id] = proba > harm_threshold
        return harmful_cache[node_id]

    def add_state(state: SearchState) -> Optional[int]:
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
            return None
        best_cost[key] = state.total_cost
        states.append(state)
        return len(states) - 1

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

    import heapq

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
                heapq.heappush(frontier, (split_child.total_cost, tie, cid))

            for child_id in nodes[state.g0].children.values():
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
                    heapq.heappush(frontier, (child.total_cost, tie, cid))
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
                    heapq.heappush(frontier, (child.total_cost, tie, cid))

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
                    heapq.heappush(frontier, (child.total_cost, tie, cid))

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
                    heapq.heappush(frontier, (child.total_cost, tie, cid))
            continue

        if harmful_streak_prune and state.harmful1_streak >= harmful_streak_prune:
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
                heapq.heappush(frontier, (child.total_cost, tie, cid))

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
                heapq.heappush(frontier, (child.total_cost, tie, cid))

    return best_goal_state_id, best_goal_cost, states


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


def main() -> None:
    args = parse_args()
    tokenizer = load_tokenizer(args.model)
    nodes, prompt_text, max_len = build_trie(tokenizer, args.jsonl, args.prompt)
    if args.max_length is not None:
        max_len = args.max_length
    if max_len <= 0:
        raise SystemExit("Max generated length must be positive.")

    harm_detector = load_harm_detector(args.detector)
    goal_id, best_cost, states = offline_search(
        nodes=nodes,
        tokenizer=tokenizer,
        harm_detector=harm_detector,
        harm_threshold=args.harm_threshold,
        harmful_streak_prune=args.harmful_streak_prune,
        max_len=max_len,
    )
    if goal_id is None:
        raise SystemExit("No goal found in offline search tree.")

    actions = reconstruct_plan(states, goal_id)
    wcd = compute_wcd(actions)
    print(f"Prompt: {prompt_text}")
    print(f"Nodes in trie: {len(nodes)} | max_len={max_len}")
    print(f"Best cost: {best_cost:.6f}")
    print(f"WCD (nodes before DoSplit): {wcd}")


if __name__ == "__main__":
    main()

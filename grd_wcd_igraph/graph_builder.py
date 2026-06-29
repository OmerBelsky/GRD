from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .igraph_compat import Graph
from .types import PrefixGraph


def _extract_continuation_ids(row: dict, tokenizer, prompt_tokens: List[int]) -> List[int]:
    continuation_ids = row.get("continuation_ids")
    if isinstance(continuation_ids, list) and continuation_ids and all(isinstance(tok, int) for tok in continuation_ids):
        return continuation_ids

    generated = (row.get("generated") or row.get("continuation") or "").strip()
    if generated:
        return tokenizer.encode(generated, add_special_tokens=False)

    full_text = row.get("full_text")
    if isinstance(full_text, str) and full_text.strip():
        full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
        if len(full_tokens) >= len(prompt_tokens) and full_tokens[: len(prompt_tokens)] == prompt_tokens:
            return full_tokens[len(prompt_tokens) :]

    return []


def build_prefix_graph(tokenizer, jsonl_path: str, prompt_override: Optional[str] = None, max_rows: Optional[int] = None) -> PrefixGraph:
    prompt_text: Optional[str] = None

    parent: List[int] = [-1]
    depth: List[int] = [0]
    token_id: List[Optional[int]] = [None]
    gen_tokens: List[Tuple[int, ...]] = [tuple()]
    visit_count: List[int] = [0]
    first_seen_row: List[int] = [-1]
    edges: List[Tuple[int, int]] = []

    path_to_vid: Dict[Tuple[int, ...], int] = {tuple(): 0}

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

            row_prompt = row.get("prompt")
            if not isinstance(row_prompt, str) or not row_prompt.strip():
                continue

            if prompt_text is None:
                prompt_text = row_prompt

            active_prompt = prompt_override or prompt_text
            prompt_tokens = tokenizer.encode(active_prompt, add_special_tokens=False)
            continuation = _extract_continuation_ids(row, tokenizer, prompt_tokens)
            if not continuation:
                continue

            current_path: List[int] = []
            for tok in continuation:
                current_path.append(tok)
                path_tuple = tuple(current_path)
                vid = path_to_vid.get(path_tuple)
                if vid is None:
                    parent_path = tuple(current_path[:-1])
                    parent_vid = path_to_vid[parent_path]
                    vid = len(parent)
                    path_to_vid[path_tuple] = vid

                    parent.append(parent_vid)
                    depth.append(len(current_path))
                    token_id.append(tok)
                    gen_tokens.append(path_tuple)
                    visit_count.append(0)
                    first_seen_row.append(row_idx)
                    edges.append((parent_vid, vid))

                visit_count[vid] += 1

    if prompt_text is None:
        raise ValueError("No valid prompt rows found in JSONL.")

    graph = Graph(n=len(parent), edges=edges, directed=True)
    graph.vs["parent"] = parent
    graph.vs["depth"] = depth
    graph.vs["token_id"] = token_id
    graph.vs["visit_count"] = visit_count
    graph.vs["first_seen_row"] = first_seen_row

    children = [graph.neighbors(vid, mode="out") for vid in range(graph.vcount())]
    leaves = [vid for vid, child_ids in enumerate(children) if vid != 0 and len(child_ids) == 0]
    leaf_set = set(leaves)
    graph.vs["is_leaf"] = [vid in leaf_set for vid in range(graph.vcount())]

    nodes_by_depth: Dict[int, List[int]] = defaultdict(list)
    for vid, d in enumerate(depth):
        nodes_by_depth[d].append(vid)

    max_depth = max(depth) if depth else 0
    return PrefixGraph(
        graph=graph,
        prompt=prompt_override or prompt_text,
        max_depth=max_depth,
        parent=parent,
        depth=depth,
        token_id=token_id,
        gen_tokens=gen_tokens,
        children=children,
        nodes_by_depth=dict(nodes_by_depth),
        leaves=leaves,
        path_to_vid=path_to_vid,
        visit_count=visit_count,
        first_seen_row=first_seen_row,
    )


def get_node_text(tokenizer, prefix_graph: PrefixGraph, node_id: int) -> str:
    return tokenizer.decode(prefix_graph.gen_tokens[node_id], skip_special_tokens=False, clean_up_tokenization_spaces=True)

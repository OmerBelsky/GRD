import argparse
import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import dill
import pandas as pd
from igraph import Graph, plot
from pyvis.network import Network

HARMFUL_NODE = "__HARMFUL__"
ENDED_NODE = "__ENDED__"


@dataclass
class PathNode:
    node_id: str
    label: str
    text: str
    scoreable: bool
    kind: str


def load_harm_detector(detector_path: str) -> Callable[[str], float]:
    with open(detector_path, "rb") as f:
        detector = dill.load(f)

    def _score(text: str) -> float:
        df = pd.DataFrame({"text": [text]})
        return float(detector.predict_proba(df)[:, 1][0])

    return _score


def truncate_text(text: str, max_len: Optional[int]) -> str:
    if max_len is None or max_len <= 0 or len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def strip_prompt_text(text: str, prompt: str) -> str:
    if prompt and text.startswith(prompt):
        return text[len(prompt) :].lstrip()
    return text


def load_tree_log(path: str) -> Tuple[Dict[int, dict], Optional[int], Optional[int]]:
    states: Dict[int, dict] = {}
    best_goal_id: Optional[int] = None
    best_goal_cost: Optional[int] = None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            state_id = row["id"]
            states[state_id] = row
            if row.get("found_harmful0") and row.get("found_no_harm1"):
                cost = row.get("total_cost")
                if cost is None:
                    continue
                if best_goal_cost is None or cost < best_goal_cost:
                    best_goal_cost = cost
                    best_goal_id = state_id
                elif cost == best_goal_cost and best_goal_id is not None and state_id < best_goal_id:
                    best_goal_id = state_id
    return states, best_goal_id, best_goal_cost


def reconstruct_path(states: Dict[int, dict], goal_id: int) -> List[dict]:
    path = []
    cursor = goal_id
    while cursor is not None:
        state = states[cursor]
        path.append(state)
        cursor = state.get("parent")
    path.reverse()
    return path


def node_for_state_action(
    state: dict,
    include_actions: bool,
) -> Optional[Tuple[str, bool, str]]:
    action = state.get("action")
    if action is None:
        return None
    if action.startswith("generate-01"):
        return state.get("g0_text", ""), True, "g0"
    if action.startswith("generate-0"):
        return state.get("g0_text", ""), True, "g0"
    if action.startswith("generate-1"):
        return state.get("g1_text", ""), True, "g1"
    if action == "mark-found-harm-0":
        return HARMFUL_NODE, False, "harmful"
    if action == "mark-found-no-harm-1":
        return ENDED_NODE, False, "ended"
    if include_actions and action in {"DoSplit", "Done0"}:
        return f"[{action}]", False, "action"
    return None


def build_path_nodes(
    path: List[dict],
    score_fn: Callable[[str], float],
    score_decimals: int,
    max_label_len: Optional[int],
    include_actions: bool,
    prompt_text: str,
) -> List[PathNode]:
    nodes: List[PathNode] = []
    score_cache: Dict[str, float] = {}

    def add_node(text: str, scoreable: bool, kind: str) -> None:
        node_id = f"n{len(nodes)}"
        label_text = truncate_text(text, max_label_len)
        if scoreable:
            if text not in score_cache:
                score_cache[text] = score_fn(text)
            score = score_cache[text]
            label = f"{label_text}\nscore={score:.{score_decimals}f}"
        else:
            label = label_text
        nodes.append(PathNode(node_id=node_id, label=label, text=text, scoreable=scoreable, kind=kind))

    if not path:
        return nodes

    add_node("(START)", False, "start")

    for state in path[1:]:
        info = node_for_state_action(state, include_actions)
        if info is None:
            continue
        text, scoreable, kind = info
        if kind in {"g0", "g1"}:
            text = strip_prompt_text(text, prompt_text)
        add_node(text, scoreable, kind)

    return nodes


def build_graph(nodes: List[PathNode], edges: Optional[List[Tuple[str, str]]] = None) -> Graph:
    g = Graph(directed=True)
    g.add_vertices([n.node_id for n in nodes])
    if edges is None:
        edges = [(nodes[i].node_id, nodes[i + 1].node_id) for i in range(len(nodes) - 1)]
    if edges:
        g.add_edges(edges)
    g.vs["label"] = [n.label for n in nodes]
    g.vs["kind"] = [n.kind for n in nodes]
    return g


def compute_layout(nodes: List[PathNode], edges: List[Tuple[str, str]]) -> Tuple[List[float], List[float]]:
    id_to_idx = {n.node_id: i for i, n in enumerate(nodes)}
    indegree = [0] * len(nodes)
    adj: Dict[int, List[int]] = {i: [] for i in range(len(nodes))}
    for src, dst in edges:
        if src not in id_to_idx or dst not in id_to_idx:
            continue
        s = id_to_idx[src]
        t = id_to_idx[dst]
        adj[s].append(t)
        indegree[t] += 1

    queue = [i for i, deg in enumerate(indegree) if deg == 0]
    topo = []
    while queue:
        v = queue.pop(0)
        topo.append(v)
        for w in adj[v]:
            indegree[w] -= 1
            if indegree[w] == 0:
                queue.append(w)

    x = [0.0 for _ in nodes]
    for v in topo:
        for w in adj[v]:
            x[w] = max(x[w], x[v] + 1.0)

    y = []
    for n in nodes:
        if n.kind in {"g0", "harmful", "g0_action"}:
            y.append(-1.0)
        elif n.kind in {"g1", "ended"}:
            y.append(1.0)
        else:
            y.append(0.0)
    return x, y


def build_branching_nodes_edges(
    path: List[dict],
    score_fn: Callable[[str], float],
    score_decimals: int,
    max_label_len: Optional[int],
    include_actions: bool,
    prompt_text: str,
) -> Tuple[List[PathNode], List[Tuple[str, str]], List[str], List[str]]:
    nodes: List[PathNode] = []
    edges: List[Tuple[str, str]] = []
    score_cache: Dict[str, float] = {}
    g0_texts: List[str] = []
    g1_texts: List[str] = []

    def add_node(text: str, scoreable: bool, kind: str) -> str:
        node_id = f"n{len(nodes)}"
        label_text = truncate_text(text, max_label_len)
        if scoreable:
            if text not in score_cache:
                score_cache[text] = score_fn(text)
            score = score_cache[text]
            label = f"{label_text}\nscore={score:.{score_decimals}f}"
        else:
            label = label_text
        nodes.append(PathNode(node_id=node_id, label=label, text=text, scoreable=scoreable, kind=kind))
        return node_id

    if not path:
        return nodes, edges, g0_texts, g1_texts

    split_idx = None
    for i, state in enumerate(path):
        if state.get("action") == "DoSplit":
            split_idx = i
            break

    if split_idx is None:
        linear_nodes = build_path_nodes(
            path,
            score_fn=score_fn,
            score_decimals=score_decimals,
            max_label_len=max_label_len,
            include_actions=include_actions,
            prompt_text=prompt_text,
        )
        linear_edges = [
            (linear_nodes[i].node_id, linear_nodes[i + 1].node_id)
            for i in range(len(linear_nodes) - 1)
        ]
        return linear_nodes, linear_edges, g0_texts, g1_texts

    last_shared_id = add_node("(START)", False, "start")

    for state in path[1:split_idx]:
        action = state.get("action") or ""
        if action.startswith("generate-01"):
            text = strip_prompt_text(state.get("g0_text", ""), prompt_text)
            node_id = add_node(text, True, "g0")
            edges.append((last_shared_id, node_id))
            last_shared_id = node_id
            g0_texts.append(text)
            g1_texts.append(text)

    do_split_id = add_node("[DoSplit]", False, "action")
    edges.append((last_shared_id, do_split_id))

    split_state = path[split_idx]
    g0_text = strip_prompt_text(split_state.get("g0_text", ""), prompt_text)
    g1_text = strip_prompt_text(split_state.get("g1_text", ""), prompt_text)
    g0_id = add_node(g0_text, True, "g0")
    g1_id = add_node(g1_text, True, "g1")
    edges.append((do_split_id, g0_id))
    edges.append((do_split_id, g1_id))
    if g0_text:
        g0_texts.append(g0_text)
    if g1_text:
        g1_texts.append(g1_text)

    last_g0_id = g0_id
    last_g1_id = g1_id

    for state in path[split_idx + 1 :]:
        action = state.get("action") or ""
        if action.startswith("generate-0"):
            text = strip_prompt_text(state.get("g0_text", ""), prompt_text)
            node_id = add_node(text, True, "g0")
            edges.append((last_g0_id, node_id))
            last_g0_id = node_id
            if text:
                g0_texts.append(text)
        elif action == "mark-found-harm-0":
            node_id = add_node(HARMFUL_NODE, False, "harmful")
            edges.append((last_g0_id, node_id))
            last_g0_id = node_id
        elif action == "Done0" and include_actions:
            node_id = add_node("[Done0]", False, "g0_action")
            edges.append((last_g0_id, node_id))
            last_g0_id = node_id

        if action.startswith("generate-1"):
            text = strip_prompt_text(state.get("g1_text", ""), prompt_text)
            node_id = add_node(text, True, "g1")
            edges.append((last_g1_id, node_id))
            last_g1_id = node_id
            if text:
                g1_texts.append(text)
        elif action == "mark-found-no-harm-1":
            node_id = add_node(ENDED_NODE, False, "ended")
            edges.append((last_g1_id, node_id))
            last_g1_id = node_id

    return nodes, edges, g0_texts, g1_texts


def plot_png(g: Graph, nodes: List[PathNode], out_png: str) -> None:
    colors = []
    shapes = []
    for n in nodes:
        if n.kind in {"harmful"}:
            colors.append("tomato")
            shapes.append("rectangle")
        elif n.kind in {"ended"}:
            colors.append("seagreen")
            shapes.append("rectangle")
        elif n.kind in {"g1"}:
            colors.append("lightskyblue")
            shapes.append("circle")
        elif n.kind in {"action", "g0_action"}:
            colors.append("lightgray")
            shapes.append("rectangle")
        elif n.kind in {"start"}:
            colors.append("gold")
            shapes.append("rectangle")
        else:
            colors.append("skyblue")
            shapes.append("circle")
    g.vs["color"] = colors
    g.vs["shape"] = shapes
    g.vs["size"] = 28
    g.vs["label_size"] = 10

    edges = [(g.vs[e.source]["name"], g.vs[e.target]["name"]) for e in g.es]
    xs, ys = compute_layout(nodes, edges)
    coords = [(xs[i] * 220.0, ys[i] * 140.0) for i in range(len(nodes))]
    width = max(800, 220 * (int(max(xs)) + 2))
    height = 450
    plot(g, out_png, layout=coords, bbox=(width, height), margin=40)


def export_html(
    g: Graph,
    nodes: List[PathNode],
    html_path: str,
    header_html: str = "",
) -> None:
    edges = [(g.vs[e.source]["name"], g.vs[e.target]["name"]) for e in g.es]
    xs_rel, ys_rel = compute_layout(nodes, edges)
    width_px = max(1200, 240 * (int(max(xs_rel)) + 2))
    height_px = 480

    min_x = min(xs_rel) if xs_rel else 0.0
    max_x = max(xs_rel) if xs_rel else 0.0
    span = max_x - min_x if max_x > min_x else 1.0
    xs = [((x - min_x) / span) * (width_px - 200) + 100 for x in xs_rel]
    ys = [height_px / 2.0 + (y * 140.0) for y in ys_rel]

    net = Network(height=f"{height_px}px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.set_options(
        json.dumps(
            {
                "physics": {"enabled": False},
                "edges": {"smooth": {"type": "straightCross"}},
                "interaction": {"hover": True, "navigationButtons": True},
            }
        )
    )

    for i, n in enumerate(nodes):
        if n.kind in {"harmful"}:
            color, shape = "tomato", "box"
        elif n.kind in {"ended"}:
            color, shape = "seagreen", "box"
        elif n.kind in {"g1"}:
            color, shape = "lightskyblue", "dot"
        elif n.kind in {"action", "g0_action"}:
            color, shape = "lightgray", "box"
        elif n.kind in {"start"}:
            color, shape = "gold", "box"
        else:
            color, shape = "skyblue", "dot"

        net.add_node(
            n_id=n.node_id,
            label=n.label,
            title=n.text,
            color=color,
            shape=shape,
            size=20 if shape == "dot" else 24,
            font={"size": 16},
            x=xs[i],
            y=ys[i],
            fixed={"x": True, "y": True},
        )

    for e in g.es:
        s = g.vs[e.source]["name"]
        t = g.vs[e.target]["name"]
        net.add_edge(s, t, arrows="to", color="rgba(60,60,60,0.95)")

    html_graph = net.generate_html(notebook=False)
    if header_html:
        html_graph = html_graph.replace("<body>", "<body>" + header_html, 1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_graph)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the optimal path from a GRD flat search tree log."
    )
    parser.add_argument(
        "--tree-log",
        required=True,
        help="Path to GRD JSONL tree log (grd_search_tree_flat*.jsonl).",
    )
    parser.add_argument(
        "--detector",
        default="harm_detector/models/binary_harm_detector.dill",
        help="Path to pickled harm detector.",
    )
    parser.add_argument(
        "--output-html",
        default="optimal_path.html",
        help="Output HTML file path.",
    )
    parser.add_argument(
        "--output-png",
        default=None,
        help="Optional PNG output path (uses igraph).",
    )
    parser.add_argument(
        "--score-decimals",
        type=int,
        default=2,
        help="Decimal places for harmfulness scores.",
    )
    parser.add_argument(
        "--max-label-len",
        type=int,
        default=120,
        help="Max label length before truncation (0 disables truncation).",
    )
    parser.add_argument(
        "--include-actions",
        action="store_true",
        help="Include Done0 action node (DoSplit is always shown when branching).",
    )
    parser.add_argument(
        "--no-branch-from-split",
        action="store_true",
        help="Render as a single linear path instead of branching at DoSplit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    states, goal_id, best_cost = load_tree_log(args.tree_log)
    if goal_id is None:
        raise SystemExit("No goal state found (found_harmful0 and found_no_harm1).")

    score_fn = load_harm_detector(args.detector)
    path = reconstruct_path(states, goal_id)
    header_html = ""
    prompt_text = path[0].get("g0_text", "") if path else ""
    if args.no_branch_from_split:
        nodes = build_path_nodes(
            path,
            score_fn=score_fn,
            score_decimals=args.score_decimals,
            max_label_len=args.max_label_len,
            include_actions=args.include_actions,
            prompt_text=prompt_text,
        )
        edges = [(nodes[i].node_id, nodes[i + 1].node_id) for i in range(len(nodes) - 1)]
    else:
        nodes, edges, g0_texts, g1_texts = build_branching_nodes_edges(
            path,
            score_fn=score_fn,
            score_decimals=args.score_decimals,
            max_label_len=args.max_label_len,
            include_actions=args.include_actions,
            prompt_text=prompt_text,
        )
        import html as htmlmod

        harmful_text = htmlmod.escape(g0_texts[-1]) if g0_texts else ""
        safe_text = htmlmod.escape(g1_texts[-1]) if g1_texts else ""
        header_html = f"""
        <div id="seq-header">
          <div class="seq-inner">
            <div class="seq-block">
              <div class="seq-title">Final harmful sequence (g0)</div>
              <div class="seq-text">{harmful_text}</div>
            </div>
            <div class="seq-block">
              <div class="seq-title">Final non-harmful sequence (g1)</div>
              <div class="seq-text">{safe_text}</div>
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
    if not nodes:
        raise SystemExit("No nodes collected for optimal path.")

    g = build_graph(nodes, edges)
    if args.output_png:
        plot_png(g, nodes, args.output_png)
        print(f"Saved PNG to: {args.output_png}")
    export_html(g, nodes, args.output_html, header_html=header_html)
    print(f"Saved HTML to: {args.output_html}")
    print(f"Best goal cost: {best_cost} | goal_id={goal_id} | nodes={len(nodes)}")


if __name__ == "__main__":
    main()

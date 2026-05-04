import json
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

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


def _style_for_kind(kind: str) -> Tuple[str, str]:
    if kind in {"harmful"}:
        return "tomato", "box"
    if kind in {"ended"}:
        return "seagreen", "box"
    if kind in {"g1"}:
        return "lightskyblue", "dot"
    if kind in {"action", "g0_action"}:
        return "lightgray", "box"
    if kind in {"start"}:
        return "gold", "box"
    return "skyblue", "dot"


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


def build_linear_nodes_from_steps(
    path_steps: List[dict],
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

    if not path_steps:
        return nodes

    add_node("(START)", False, "start")

    for state in path_steps:
        action = state.get("action")
        if action is None:
            continue
        if action.startswith("generate-01") or action.startswith("generate-0"):
            text = strip_prompt_text(state.get("g0_text", ""), prompt_text)
            add_node(text, True, "g0")
        elif action.startswith("generate-1"):
            text = strip_prompt_text(state.get("g1_text", ""), prompt_text)
            add_node(text, True, "g1")
        elif action == "mark-found-harm-0":
            add_node(HARMFUL_NODE, False, "harmful")
        elif action == "mark-found-no-harm-1":
            add_node(ENDED_NODE, False, "ended")
        elif include_actions and action in {"DoSplit", "Done0"}:
            add_node(f"[{action}]", False, "action")

    return nodes


def build_branching_nodes_edges_from_steps(
    path_steps: List[dict],
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

    if not path_steps:
        return nodes, edges, g0_texts, g1_texts

    split_idx = None
    for i, state in enumerate(path_steps):
        if state.get("action") == "DoSplit":
            split_idx = i
            break

    if split_idx is None:
        linear_nodes = build_linear_nodes_from_steps(
            path_steps=path_steps,
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

    for state in path_steps[:split_idx]:
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

    split_state = path_steps[split_idx]
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

    for state in path_steps[split_idx + 1 :]:
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


def export_html_graph(
    nodes: List[PathNode],
    edges: List[Tuple[str, str]],
    html_path: str,
    header_html: str = "",
) -> None:
    xs_rel, ys_rel = compute_layout(nodes, edges)
    width_px = max(1200, 240 * (int(max(xs_rel)) + 2)) if xs_rel else 1200
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
        color, shape = _style_for_kind(n.kind)
        net.add_node(
            n_id=n.node_id,
            label=n.label,
            title=n.text,
            color=color,
            shape=shape,
            size=20 if shape == "dot" else 24,
            font={"size": 16},
            x=xs[i] if i < len(xs) else 0,
            y=ys[i] if i < len(ys) else 0,
            fixed={"x": True, "y": True},
        )

    for src, dst in edges:
        net.add_edge(src, dst, arrows="to", color="rgba(60,60,60,0.95)")

    html_graph = net.generate_html(notebook=False)
    if header_html:
        html_graph = html_graph.replace("<body>", "<body>" + header_html, 1)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_graph)


def export_png_graph(nodes: List[PathNode], edges: List[Tuple[str, str]], png_path: str) -> None:
    g = Graph(directed=True)
    g.add_vertices([n.node_id for n in nodes])
    if edges:
        g.add_edges(edges)
    g.vs["label"] = [n.label for n in nodes]

    colors = []
    shapes = []
    for n in nodes:
        color, shape = _style_for_kind(n.kind)
        colors.append(color)
        shapes.append("rectangle" if shape == "box" else "circle")
    g.vs["color"] = colors
    g.vs["shape"] = shapes
    g.vs["size"] = 28
    g.vs["label_size"] = 10

    xs_rel, ys_rel = compute_layout(nodes, edges)
    coords = [(xs_rel[i] * 220.0, ys_rel[i] * 140.0) for i in range(len(nodes))]
    width = max(800, 220 * (int(max(xs_rel)) + 2)) if xs_rel else 800
    height = 450
    plot(g, png_path, layout=coords, bbox=(width, height), margin=40)


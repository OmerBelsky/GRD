import argparse
import html as htmlmod
import json
import os
from typing import Dict, List, Optional, Tuple

from utils.harm import harm_proba, load_harm_detector
from utils.path_graph import (
    build_branching_nodes_edges_from_steps,
    build_linear_nodes_from_steps,
    export_html_graph,
    export_png_graph,
)


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
        default="outputs/graphs/optimal/optimal_path.html",
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

    detector = load_harm_detector(args.detector)
    score_fn = lambda text: harm_proba(detector, text)
    path = reconstruct_path(states, goal_id)
    header_html = ""
    prompt_text = path[0].get("g0_text", "") if path else ""

    if args.no_branch_from_split:
        nodes = build_linear_nodes_from_steps(
            path_steps=path,
            score_fn=score_fn,
            score_decimals=args.score_decimals,
            max_label_len=args.max_label_len,
            include_actions=args.include_actions,
            prompt_text=prompt_text,
        )
        edges = [(nodes[i].node_id, nodes[i + 1].node_id) for i in range(len(nodes) - 1)]
    else:
        nodes, edges, g0_texts, g1_texts = build_branching_nodes_edges_from_steps(
            path_steps=path,
            score_fn=score_fn,
            score_decimals=args.score_decimals,
            max_label_len=args.max_label_len,
            include_actions=args.include_actions,
            prompt_text=prompt_text,
        )

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

    if args.output_html:
        out_html_dir = os.path.dirname(args.output_html)
        if out_html_dir:
            os.makedirs(out_html_dir, exist_ok=True)
    if args.output_png:
        out_png_dir = os.path.dirname(args.output_png)
        if out_png_dir:
            os.makedirs(out_png_dir, exist_ok=True)

    if args.output_png:
        export_png_graph(nodes, edges, args.output_png)
        print(f"Saved PNG to: {args.output_png}")
    export_html_graph(nodes, edges, args.output_html, header_html=header_html)
    print(f"Saved HTML to: {args.output_html}")
    print(f"Best goal cost: {best_cost} | goal_id={goal_id} | nodes={len(nodes)}")


if __name__ == "__main__":
    main()


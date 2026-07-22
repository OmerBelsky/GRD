from __future__ import annotations

import logging
import math
import os
import random
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

from .graph_builder import build_prefix_graph, get_node_text
from .harm_labeling import assign_goal_labels, score_harm_probabilities
from .interventions import build_intervention_selection, load_batch_specs, spec_from_args
from .mass import compute_mass_result
from .propagation import compute_reachable_nodes, compute_wcd, propagate_labels
from .reporting import build_report_payload, write_report


@contextmanager
def stage_timer(stage_name: str, timings: Dict[str, float], logger: logging.Logger):
    start = time.perf_counter()
    logger.info("Starting stage: %s", stage_name)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        timings[stage_name] = elapsed
        logger.info("Finished stage: %s in %.2fs", stage_name, elapsed)


def _mass_of_nodes(node_log_mass: List[float], node_ids: List[int]) -> float:
    total = 0.0
    for node_id in node_ids:
        value = node_log_mass[node_id]
        if value == -float("inf"):
            continue
        total += math.exp(value)
    return float(total)


def _intervention_slug(intervention_payload: dict) -> str:
    name = str(intervention_payload.get("intervention", "none"))
    if name == "none":
        return "intervention=none"
    k = intervention_payload.get("k")
    top_n = intervention_payload.get("top_n")
    selection = intervention_payload.get("selection")
    return f"intervention={name}__k={k}__top_n={top_n}__selection={selection}"


def _report_filename_for_intervention(base_filename: str | None, intervention_payload: dict, total_specs: int) -> str | None:
    if base_filename is None:
        return None
    if total_specs <= 1:
        return base_filename
    stem, ext = os.path.splitext(base_filename)
    if not ext:
        ext = ".json"
    return f"{stem}__{_intervention_slug(intervention_payload)}{ext}"


def _build_parent_chain(prefix_graph, node_id: int) -> List[int]:
    chain: List[int] = []
    cursor = node_id
    while cursor > 0:
        chain.append(cursor)
        cursor = prefix_graph.parent[cursor]
    chain.reverse()
    return chain


def _find_descendant_goal(
    *,
    prefix_graph,
    start_node: int,
    reachable: List[bool],
    target_goal_flags: List[bool],
    rng: random.Random,
) -> Optional[int]:
    stack = [start_node]
    candidates: List[int] = []
    while stack:
        node = stack.pop()
        if not reachable[node]:
            continue
        if target_goal_flags[node]:
            candidates.append(node)
        for child_id in prefix_graph.children[node]:
            if reachable[child_id]:
                stack.append(child_id)
    if not candidates:
        return None
    return rng.choice(candidates)


def _build_graph_steps(
    *,
    prefix_graph,
    tokenizer,
    split_node: int,
    harmful_goal_node: int,
    safe_goal_node: int,
) -> List[dict]:
    shared_chain = _build_parent_chain(prefix_graph, split_node)
    harmful_chain = _build_parent_chain(prefix_graph, harmful_goal_node)
    safe_chain = _build_parent_chain(prefix_graph, safe_goal_node)

    shared_set = set(shared_chain)
    harmful_suffix = [node for node in harmful_chain if node not in shared_set]
    safe_suffix = [node for node in safe_chain if node not in shared_set]

    steps: List[dict] = []
    for node in shared_chain:
        steps.append(
            {
                "action": "generate-01",
                "g0_text": get_node_text(tokenizer, prefix_graph, node),
                "g1_text": get_node_text(tokenizer, prefix_graph, node),
            }
        )

    steps.append(
        {
            "action": "DoSplit",
            "g0_text": get_node_text(tokenizer, prefix_graph, split_node),
            "g1_text": get_node_text(tokenizer, prefix_graph, split_node),
        }
    )

    for idx in range(max(len(harmful_suffix), len(safe_suffix))):
        if idx < len(harmful_suffix):
            node = harmful_suffix[idx]
            steps.append(
                {
                    "action": "generate-0",
                    "g0_text": get_node_text(tokenizer, prefix_graph, node),
                }
            )
        if idx < len(safe_suffix):
            node = safe_suffix[idx]
            steps.append(
                {
                    "action": "generate-1",
                    "g1_text": get_node_text(tokenizer, prefix_graph, node),
                }
            )

    steps.append({"action": "mark-found-harm-0", "g0_text": get_node_text(tokenizer, prefix_graph, harmful_goal_node)})
    steps.append({"action": "Done0", "g0_text": get_node_text(tokenizer, prefix_graph, harmful_goal_node)})
    steps.append({"action": "mark-found-no-harm-1", "g1_text": get_node_text(tokenizer, prefix_graph, safe_goal_node)})
    return steps


def _score_text_stub(_: str) -> float:
    return 0.0


def _write_spec_graph_html(
    *,
    args,
    prefix_graph,
    tokenizer,
    intervention_payload: dict,
    report_path: str,
    reachable: List[bool],
    ambiguous: List[bool],
    harmful_goal_flags: List[bool],
    safe_goal_flags: List[bool],
) -> Optional[str]:
    from utils.path_graph import build_branching_nodes_edges_from_steps, export_html_graph

    split_candidates = [idx for idx, is_ambiguous in enumerate(ambiguous) if is_ambiguous and reachable[idx]]
    if not split_candidates:
        return None

    rng = random.Random(0)
    split_node = rng.choice(split_candidates)
    harmful_goal = _find_descendant_goal(
        prefix_graph=prefix_graph,
        start_node=split_node,
        reachable=reachable,
        target_goal_flags=harmful_goal_flags,
        rng=rng,
    )
    safe_goal = _find_descendant_goal(
        prefix_graph=prefix_graph,
        start_node=split_node,
        reachable=reachable,
        target_goal_flags=safe_goal_flags,
        rng=rng,
    )
    if harmful_goal is None or safe_goal is None:
        return None

    steps = _build_graph_steps(
        prefix_graph=prefix_graph,
        tokenizer=tokenizer,
        split_node=split_node,
        harmful_goal_node=harmful_goal,
        safe_goal_node=safe_goal,
    )

    nodes, edges, _, _ = build_branching_nodes_edges_from_steps(
        path_steps=steps,
        score_fn=_score_text_stub,
        score_decimals=2,
        max_label_len=None,
        include_actions=True,
        prompt_text="",
    )

    os.makedirs(args.graph_dir, exist_ok=True)
    html_filename = os.path.basename(os.path.splitext(report_path)[0]) + ".html"
    html_path = os.path.join(args.graph_dir, html_filename)
    export_html_graph(nodes, edges, html_path)
    return html_path


def run_pipeline(args) -> List[str]:
    logger = logging.getLogger("grd_wcd_igraph")
    timings: Dict[str, float] = {}
    overall_start = time.perf_counter()

    from utils.harm import load_harm_detector
    from utils.modeling import load_model_auto, load_tokenizer

    with stage_timer("load_tokenizer", timings, logger):
        tokenizer = load_tokenizer(args.model, hf_token=args.hf_token)

    model = None
    vllm_client = None
    if args.model_backend == "transformers":
        with stage_timer("load_model", timings, logger):
            _, model = load_model_auto(
                args.model,
                args.hf_token,
                device=args.model_device,
            )
    else:
        from grd_generation_mass import VLLMGenerateClient

        vllm_client = VLLMGenerateClient(
            endpoint=args.vllm_endpoint,
            model_name=args.model,
            vocab_token_ids=list(range(len(tokenizer))),
        )

    with stage_timer("build_graph", timings, logger):
        prefix_graph = build_prefix_graph(
            tokenizer=tokenizer,
            jsonl_path=args.jsonl,
            prompt_override=args.prompt,
            max_rows=args.max_rows,
        )
        logger.info(
            "Graph built with %d nodes, %d leaves, max depth %d",
            len(prefix_graph.depth),
            len(prefix_graph.leaves),
            prefix_graph.max_depth,
        )

    with stage_timer("compute_mass", timings, logger):
        mass_result = compute_mass_result(
            prefix_graph=prefix_graph,
            tokenizer=tokenizer,
            model=model,
            backend=args.model_backend,
            vllm_client=vllm_client,
            jsonl_path=args.jsonl,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_rows=args.max_rows,
            vllm_concurrency=args.vllm_concurrency,
        )

    with stage_timer("load_harm_detector", timings, logger):
        harm_detector = load_harm_detector(args.detector)

    with stage_timer("score_harm", timings, logger):
        harm_probs = score_harm_probabilities(
            prefix_graph=prefix_graph,
            tokenizer=tokenizer,
            harm_detector=harm_detector,
            harm_start_depth=args.harm_start_depth,
            batch_size=args.harm_detector_batch_size,
        )

    with stage_timer("assign_goals", timings, logger):
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        goal_labels = assign_goal_labels(
            prefix_graph=prefix_graph,
            harm_probs=harm_probs,
            harm_threshold=args.harm_threshold,
            harm_start_depth=args.harm_start_depth,
            eos_token_id=eos_token_id,
        )
        logger.info(
            "Goal labels: harmful=%d safe=%d",
            goal_labels.harmful_goal_count,
            goal_labels.safe_goal_count,
        )

    with stage_timer("baseline_propagation", timings, logger):
        baseline_propagation = propagate_labels(
            prefix_graph=prefix_graph,
            harmful_goal=goal_labels.harmful_goal,
            safe_goal=goal_labels.safe_goal,
        )
        baseline_reachable = compute_reachable_nodes(prefix_graph=prefix_graph)
        baseline_wcd, baseline_witness = compute_wcd(
            prefix_graph,
            baseline_propagation,
            reachable=baseline_reachable,
        )

    if args.batch_specs_file:
        specs = load_batch_specs(args.batch_specs_file)
    else:
        specs = [spec_from_args(args)]

    if args.intervention_seed is not None:
        for spec in specs:
            if spec.intervention_seed is None:
                spec.intervention_seed = args.intervention_seed

    intervention_results: List[dict] = []
    with stage_timer("interventions", timings, logger):
        for spec in specs:
            logger.info(
                "Processing spec intervention=%s k=%s top_n=%s selection=%s seed=%s",
                spec.intervention,
                spec.intervention_k,
                spec.intervention_top_n,
                spec.intervention_selection,
                spec.intervention_seed,
            )
            if spec.intervention == "none":
                payload = {
                    "intervention": "none",
                    "k": None,
                    "top_n": None,
                    "selection": None,
                    "wcd": baseline_wcd,
                    "witness_node": baseline_witness,
                    "candidate_mass_nucleus_total": 0.0,
                    "selected_mass_nucleus_total": 0.0,
                    "pruned_mass_nucleus_total": 0.0,
                    "mass_nucleus_after_intervention": float(mass_result.covered_mass_nucleus),
                    "activation_count": 0,
                    "events": [],
                }
                intervention_results.append(
                    {
                        "payload": payload,
                        "reachable": baseline_reachable,
                        "ambiguous": baseline_propagation.ambiguous,
                    }
                )
                continue

            selection = build_intervention_selection(
                prefix_graph=prefix_graph,
                spec=spec,
                harm_probs=harm_probs,
            )

            propagation = propagate_labels(
                prefix_graph=prefix_graph,
                harmful_goal=goal_labels.harmful_goal,
                safe_goal=goal_labels.safe_goal,
                allowed_children_by_parent=selection.allowed_children_by_parent,
                intervention_depth=spec.intervention_k,
                baseline=baseline_propagation,
            )
            reachable = compute_reachable_nodes(
                prefix_graph=prefix_graph,
                allowed_children_by_parent=selection.allowed_children_by_parent,
                intervention_depth=spec.intervention_k,
            )
            intervention_wcd, intervention_witness = compute_wcd(prefix_graph, propagation, reachable=reachable)

            candidate_ids: List[int] = []
            selected_ids: List[int] = []
            for event in selection.events:
                candidate_ids.extend(event.candidate_child_ids)
                selected_ids.extend(event.selected_child_ids)

            candidate_mass = _mass_of_nodes(mass_result.node_log_mass_nucleus, candidate_ids)
            selected_mass = _mass_of_nodes(mass_result.node_log_mass_nucleus, selected_ids)
            pruned_mass = max(0.0, candidate_mass - selected_mass)
            mass_after = max(0.0, float(mass_result.covered_mass_nucleus) - pruned_mass)

            payload = {
                "intervention": spec.intervention,
                "k": spec.intervention_k,
                "top_n": spec.intervention_top_n,
                "selection": spec.intervention_selection,
                "seed": spec.intervention_seed,
                "wcd": intervention_wcd,
                "witness_node": intervention_witness,
                "candidate_mass_nucleus_total": candidate_mass,
                "selected_mass_nucleus_total": selected_mass,
                "pruned_mass_nucleus_total": pruned_mass,
                "mass_nucleus_after_intervention": mass_after,
                "activation_count": len(selection.events),
                "events": [
                    {
                        "parent_id": event.parent_id,
                        "depth": event.depth,
                        "candidate_child_ids": event.candidate_child_ids,
                        "selected_child_ids": event.selected_child_ids,
                        "chosen_child_id": event.chosen_child_id,
                        "chosen_harm_probability": event.chosen_harm_probability,
                    }
                    for event in selection.events
                ],
            }
            intervention_results.append(
                {
                    "payload": payload,
                    "reachable": reachable,
                    "ambiguous": propagation.ambiguous,
                }
            )

    report_paths: List[str] = []
    with stage_timer("write_report", timings, logger):
        total_specs = len(intervention_results)
        for result in intervention_results:
            intervention_payload = result["payload"]
            report_filename = _report_filename_for_intervention(args.report_filename, intervention_payload, total_specs)
            payload = build_report_payload(
                status="ok",
                jsonl=args.jsonl,
                prompt=prefix_graph.prompt,
                system_prompt_id=args.system_prompt_id,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                harm_threshold=args.harm_threshold,
                harm_start_depth=args.harm_start_depth,
                mass_summary=mass_result.mass_summary,
                baseline_wcd=baseline_wcd,
                baseline_witness_node=baseline_witness,
                harmful_goal_count=goal_labels.harmful_goal_count,
                safe_goal_count=goal_labels.safe_goal_count,
                interventions=[intervention_payload],
                runtime_seconds=time.perf_counter() - overall_start,
                timings=timings,
            )
            report_path = write_report(payload, report_dir=args.report_dir, report_filename=report_filename)

            graph_path = _write_spec_graph_html(
                args=args,
                prefix_graph=prefix_graph,
                tokenizer=tokenizer,
                intervention_payload=intervention_payload,
                report_path=report_path,
                reachable=result["reachable"],
                ambiguous=result["ambiguous"],
                harmful_goal_flags=goal_labels.harmful_goal,
                safe_goal_flags=goal_labels.safe_goal,
            )
            if graph_path is not None:
                payload["graph_outputs"] = {"html": graph_path}
                write_report(payload, report_dir=args.report_dir, report_filename=os.path.basename(report_path))

            report_paths.append(report_path)
            logger.info("Wrote report: %s", report_path)

    return report_paths

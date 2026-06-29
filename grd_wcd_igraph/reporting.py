from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_timestamp_slug() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')}_{uuid.uuid4().hex[:8]}"


def build_report_payload(
    *,
    status: str,
    jsonl: str,
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    harm_threshold: float,
    harm_start_depth: int,
    mass_summary: Dict[str, Any],
    baseline_wcd: int,
    baseline_witness_node: Any,
    harmful_goal_count: int,
    safe_goal_count: int,
    interventions: List[Dict[str, Any]],
    runtime_seconds: float,
    timings: Dict[str, float],
    graph_outputs: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "jsonl": jsonl,
        "prompt": prompt,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "harm_threshold": harm_threshold,
        "harm_start_depth": harm_start_depth,
        "mass": dict(mass_summary),
        "baseline": {
            "wcd": baseline_wcd,
            "witness_node": baseline_witness_node,
            "harmful_goal_count": harmful_goal_count,
            "safe_goal_count": safe_goal_count,
        },
        "interventions": interventions,
        "timings_seconds": dict(timings),
        "runtime_seconds": runtime_seconds,
    }
    if graph_outputs is not None:
        payload["graph_outputs"] = dict(graph_outputs)
    return payload


def write_report(payload: Dict[str, Any], report_dir: str, report_filename: str | None = None) -> str:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = report_filename or f"offline_wcd_igraph_result_{utc_timestamp_slug()}.json"
    out_path = out_dir / filename
    with out_path.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=True)
    return str(out_path)

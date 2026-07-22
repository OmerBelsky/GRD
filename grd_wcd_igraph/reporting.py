from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def utc_timestamp_slug() -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')}_{uuid.uuid4().hex[:8]}"


def _truncate_to_max_bytes(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value

    # Keep trimming until the UTF-8 byte size fits the budget.
    end = len(value)
    while end > 0:
        candidate = value[:end]
        if len(candidate.encode("utf-8")) <= max_bytes:
            return candidate
        end -= 1
    return ""


def _safe_report_filename(filename: str, max_bytes: int = 240) -> str:
    # Worst-case intervention slug: "__intervention=fixed_k__k=40__top_n=3__selection=random" = 57 bytes
    WORST_CASE_IV_BYTES = 57

    stem, ext = os.path.splitext(filename)
    if not ext:
        ext = ".json"

    intervention_sep = "__intervention="
    model_sep = "__model="

    # Only apply structured truncation when filename has known structure.
    if intervention_sep in stem:
        base_part, intervention_part = stem.split(intervention_sep, 1)
        intervention_full = intervention_sep + intervention_part

        if model_sep in base_part:
            prompt_part, meta_part = base_part.split(model_sep, 1)
            meta_full = model_sep + meta_part

            # Always budget against worst-case intervention length so every spec
            # for the same prompt truncates the prompt slug to the same length,
            # regardless of whether the actual intervention slug is shorter.
            budget = max_bytes - len(ext.encode("utf-8")) - WORST_CASE_IV_BYTES - len(meta_full.encode("utf-8"))
            budget = max(20, budget)

            if len(prompt_part.encode("utf-8")) > budget:
                truncated_prompt = _truncate_to_max_bytes(prompt_part, budget) or "report"
                return f"{truncated_prompt}{meta_full}{intervention_full}{ext}"
            # Prompt slug fits within worst-case budget → filename fits ≤ 240 too.
            return filename

    if len(filename.encode("utf-8")) <= max_bytes:
        return filename

    # Fallback: hash-based truncation for unexpected filename structures.
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]
    suffix = f"__{digest}"
    budget_for_stem = max_bytes - len(ext.encode("utf-8")) - len(suffix.encode("utf-8"))
    truncated_stem = _truncate_to_max_bytes(stem, budget_for_stem) or "report"
    return f"{truncated_stem}{suffix}{ext}"


def build_report_payload(
    *,
    status: str,
    jsonl: str,
    prompt: str,
    system_prompt_id: str | None,
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
    if system_prompt_id:
        payload["system_prompt_id"] = system_prompt_id
    if graph_outputs is not None:
        payload["graph_outputs"] = dict(graph_outputs)
    return payload


def write_report(payload: Dict[str, Any], report_dir: str, report_filename: str | None = None) -> str:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = report_filename or f"offline_wcd_igraph_result_{utc_timestamp_slug()}.json"
    out_path = out_dir / _safe_report_filename(filename)
    try:
        with out_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=True)
    except OSError as exc:
        # Defensive fallback for path length errors on stricter filesystems.
        if getattr(exc, "errno", None) != 36:
            raise
        fallback_name = f"offline_wcd_igraph_result_{utc_timestamp_slug()}.json"
        out_path = out_dir / fallback_name
        with out_path.open("w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=True)
    return str(out_path)

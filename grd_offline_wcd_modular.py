import argparse
from array import array
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import heapq
import html as htmlmod
import json
import os
import resource
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Set, Union
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> None:
        return None
from transformers import AutoTokenizer

from grd_generation_mass import load_unique_generations, logaddexp_scalar, safe_exp, score_sequence_logs
from utils.harm import harm_proba, harm_proba_batch, harm_proba_cached, is_harmful_cached, load_harm_detector
from utils.modeling import load_model_auto, load_tokenizer
from grd_generation_mass import VLLMGenerateClient
from utils.path_graph import build_branching_nodes_edges_from_steps, export_html_graph, export_png_graph

load_dotenv()

RESULTS_DIR = Path("outputs/reports/offline_wcd")
RESULTS_NAME_PREFIX = "offline_wcd_result_"
RESULTS_NAME_SUFFIX = ".json"
NODE_TEXTS: Optional[List[str]] = None


@dataclass
class TrieNode:
    parent: Optional[int]
    token_id: Optional[int]
    children: Dict[int, int]
    depth: int
    gen_tokens: Tuple[int, ...]


@dataclass(slots=True)
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


STATE_FLAG_SPLIT = 1 << 0
STATE_FLAG_DONE0 = 1 << 1
STATE_FLAG_FOUND_HARMFUL0 = 1 << 2
STATE_FLAG_FOUND_NO_HARM1 = 1 << 3

ACTION_NAME_TO_CODE = {
    None: 0,
    "generate-01": 1,
    "generate-0": 2,
    "generate-1": 3,
    "DoSplit": 4,
    "Done0": 5,
    "mark-found-harm-0": 6,
    "mark-found-no-harm-1": 7,
}
ACTION_CODE_TO_NAME = {code: name for name, code in ACTION_NAME_TO_CODE.items()}

DEFAULT_SEED_BEAM_WIDTH = 8
DEFAULT_SEED_MAX_VISITS = 2048


def pack_state_flags(
    *,
    split: bool,
    done0: bool,
    found_harmful0: bool,
    found_no_harm1: bool,
) -> int:
    flags = 0
    if split:
        flags |= STATE_FLAG_SPLIT
    if done0:
        flags |= STATE_FLAG_DONE0
    if found_harmful0:
        flags |= STATE_FLAG_FOUND_HARMFUL0
    if found_no_harm1:
        flags |= STATE_FLAG_FOUND_NO_HARM1
    return flags


class SearchStateStore:
    __slots__ = (
        "g0_ids",
        "g1_ids",
        "flags",
        "harmful1_streaks",
        "total_costs",
        "parents",
        "actions",
    )

    def __init__(self) -> None:
        self.g0_ids = array("I")
        self.g1_ids = array("I")
        self.flags = array("B")
        self.harmful1_streaks = array("I")
        self.total_costs = array("d")
        self.parents = array("q")
        self.actions = array("B")

    def __len__(self) -> int:
        return len(self.g0_ids)

    def add(
        self,
        *,
        g0: int,
        g1: int,
        flags: int,
        harmful1_streak: int,
        total_cost: float,
        parent: Optional[int],
        action: Optional[str],
    ) -> int:
        self.g0_ids.append(g0)
        self.g1_ids.append(g1)
        self.flags.append(flags)
        self.harmful1_streaks.append(harmful1_streak)
        self.total_costs.append(total_cost)
        self.parents.append(-1 if parent is None else parent)
        self.actions.append(ACTION_NAME_TO_CODE[action])
        return len(self.g0_ids) - 1

    def g0(self, state_id: int) -> int:
        return self.g0_ids[state_id]

    def g1(self, state_id: int) -> int:
        return self.g1_ids[state_id]

    def total_cost(self, state_id: int) -> float:
        return self.total_costs[state_id]

    def parent(self, state_id: int) -> Optional[int]:
        parent_id = self.parents[state_id]
        return None if parent_id < 0 else parent_id

    def action(self, state_id: int) -> Optional[str]:
        return ACTION_CODE_TO_NAME[self.actions[state_id]]

    def harmful1_streak(self, state_id: int) -> int:
        return self.harmful1_streaks[state_id]

    def has_flag(self, state_id: int, mask: int) -> bool:
        return bool(self.flags[state_id] & mask)

    def split(self, state_id: int) -> bool:
        return self.has_flag(state_id, STATE_FLAG_SPLIT)

    def done0(self, state_id: int) -> bool:
        return self.has_flag(state_id, STATE_FLAG_DONE0)

    def found_harmful0(self, state_id: int) -> bool:
        return self.has_flag(state_id, STATE_FLAG_FOUND_HARMFUL0)

    def found_no_harm1(self, state_id: int) -> bool:
        return self.has_flag(state_id, STATE_FLAG_FOUND_NO_HARM1)

    def snapshot(self, state_id: int) -> SearchState:
        return SearchState(
            g0=self.g0(state_id),
            g1=self.g1(state_id),
            split=self.split(state_id),
            done0=self.done0(state_id),
            found_harmful0=self.found_harmful0(state_id),
            found_no_harm1=self.found_no_harm1(state_id),
            harmful1_streak=self.harmful1_streak(state_id),
            total_cost=self.total_cost(state_id),
            parent=self.parent(state_id),
            action=self.action(state_id),
        )


@dataclass
class InterventionEvent:
    policy: str
    trigger: str
    state_id: int
    depth: int
    action_context: str
    chosen_child_id: int
    selected_child_ids: List[int]
    candidate_child_ids: List[int]
    chosen_harm_probability: float


@dataclass
class MassSummary:
    rows_read: int
    rows_skipped_invalid: int
    rows_empty_continuation: int
    rows_skipped_empty_prompt: int
    rows_duplicates: int
    used_json_continuation_ids: int
    used_full_text_tokenization: int
    used_separate_tokenization: int
    unique_generations: int
    total_continuation_tokens_scored: int
    covered_mass_original: float
    covered_mass_nucleus: float
    covered_log_mass_original: float
    covered_log_mass_nucleus: float
    node_log_mass_original: Union[List[float], Dict[int, float]]
    node_log_mass_nucleus: Union[List[float], Dict[int, float]]


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
        harm_score_cache: Optional[Dict[int, float]] = None,
    ) -> Tuple[List[int], Optional[InterventionEvent]]:
        return child_ids, None


class NoInterventionPolicy(InterventionPolicy):
    name = "none"


class FixedKForceTokenPolicy(InterventionPolicy):
    """
    Trie-only adaptation of fixed-k forcing used in intervention generation.

    Supports two selection modes:
    - extreme: keep the top_n candidates maximizing |p(harm) - 0.5|
    - both_sides: keep the top_n highest and top_n lowest harm probabilities
    """

    name = "fixed_k_force_token"

    def __init__(
        self,
        trigger: InterventionTrigger,
        top_n: int,
        selection_mode: str,
        harm_detector_batch_size: int,
    ):
        self.trigger = trigger
        self.top_n = max(1, top_n)
        self.selection_mode = selection_mode
        self.harm_detector_batch_size = max(1, int(harm_detector_batch_size))
        self.name = f"fixed_k_force_token_{selection_mode}_top_{self.top_n}"

    def _score_candidates(
        self,
        child_ids: List[int],
        trie_nodes: List[TrieNode],
        tokenizer,
        harm_detector,
        harm_score_cache: Optional[Dict[int, float]] = None,
    ) -> List[Tuple[int, float]]:
        return score_child_ids_harm(
            child_ids=child_ids,
            trie_nodes=trie_nodes,
            tokenizer=tokenizer,
            harm_detector=harm_detector,
            harm_score_cache=harm_score_cache,
            batch_size=self.harm_detector_batch_size,
        )

    def _select_extreme(self, scored_children: List[Tuple[int, float]]) -> List[int]:
        ranked = sorted(
            scored_children,
            key=lambda item: (abs(item[1] - 0.5), item[1]),
            reverse=True,
        )
        return [child_id for child_id, _ in ranked[: self.top_n]]

    def _select_both_sides(self, scored_children: List[Tuple[int, float]]) -> List[int]:
        ranked_high = sorted(scored_children, key=lambda item: (item[1], -item[0]), reverse=True)
        ranked_low = sorted(scored_children, key=lambda item: (item[1], item[0]))

        selected: List[int] = []
        for child_id, _ in ranked_high[: self.top_n]:
            if child_id not in selected:
                selected.append(child_id)
        for child_id, _ in ranked_low[: self.top_n]:
            if child_id not in selected:
                selected.append(child_id)
        return selected

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
        harm_score_cache: Optional[Dict[int, float]] = None,
    ) -> Tuple[List[int], Optional[InterventionEvent]]:
        if not self.trigger.should_trigger(depth, action_context, state):
            return child_ids, None

        if not child_ids:
            return child_ids, None

        scored_children = self._score_candidates(
            child_ids,
            trie_nodes,
            tokenizer,
            harm_detector,
            harm_score_cache=harm_score_cache,
        )
        if not scored_children:
            return child_ids, None

        if self.selection_mode == "extreme":
            selected_child_ids = self._select_extreme(scored_children)
        elif self.selection_mode == "both_sides":
            selected_child_ids = self._select_both_sides(scored_children)
        else:
            raise ValueError(f"Unknown selection mode: {self.selection_mode}")

        if not selected_child_ids:
            return child_ids, None

        chosen_child_id = selected_child_ids[0]
        chosen_harm_probability = next(
            (p for child_id, p in scored_children if child_id == chosen_child_id),
            0.0,
        )

        event = InterventionEvent(
            policy=self.name,
            trigger=self.trigger.name,
            state_id=state_id,
            depth=depth,
            action_context=action_context,
            chosen_child_id=chosen_child_id,
            selected_child_ids=selected_child_ids,
            candidate_child_ids=list(child_ids),
            chosen_harm_probability=chosen_harm_probability,
        )
        return selected_child_ids, event


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
        help="HuggingFace model identifier used for mass scoring.",
    )
    parser.add_argument(
        "--model-backend",
        type=str,
        default="transformers",
        choices=["transformers", "vllm"],
        help="Backend used to score generation mass.",
    )
    parser.add_argument(
        "--vllm-endpoint",
        type=str,
        default=os.getenv("VLLM_GENERATE_ENDPOINT", "http://127.0.0.1:8000/inference/v1/generate"),
        help="vLLM token-in/token-out generate endpoint for backend=vllm.",
    )
    parser.add_argument(
        "--vllm-concurrency",
        type=int,
        default=8,
        help="Maximum number of in-flight vLLM scoring requests to run concurrently.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (or HF_TOKEN env var).",
    )
    parser.add_argument(
        "--model-device",
        type=str,
        default=None,
        help=(
            "Force model to a specific device (e.g. cpu, cuda, cuda:0). "
            "If unset, loader falls back to GRD_MODEL_DEVICE env var or device_map behavior."
        ),
    )
    parser.add_argument(
        "--single-free-gpu",
        action="store_true",
        help="Place model on exactly one CUDA device with the most currently free VRAM.",
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
        "--temperature",
        type=float,
        default=0.5,
        help="Sampling temperature used to score generation mass.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus top-p used to score generation mass.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k used to score generation mass (0 disables).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of JSONL rows to score for mass accounting.",
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
        "--seed-goal-bound",
        action="store_true",
        help="Compute an initial complete-solution upper bound before the main search loop.",
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
        default=None,
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
    parser.add_argument(
        "--intervention-top-n",
        type=int,
        default=1,
        help="Number of tokens to keep when the intervention triggers.",
    )
    parser.add_argument(
        "--intervention-selection",
        type=str,
        default="extreme",
        choices=["extreme", "both_sides"],
        help="How to choose the kept tokens when the intervention triggers.",
    )
    parser.add_argument(
        "--harm-detector-batch-size",
        type=int,
        default=128,
        help="Batch size for harm-detector probability scoring.",
    )
    parser.add_argument(
        "--log-memory-checkpoints",
        action="store_true",
        help="Log host-memory checkpoints around major offline WCD phases.",
    )
    parser.add_argument(
        "--memory-log-file",
        type=str,
        default=None,
        help="Optional file path that receives memory checkpoint and in-search memory logs.",
    )
    parser.add_argument(
        "--search-memory-report-interval",
        type=int,
        default=100000,
        help="Expanded-state interval for periodic in-search memory reporting (0 disables periodic search logs).",
    )
    parser.add_argument(
        "--batch-specs-file",
        type=str,
        default=None,
        help=(
            "Optional JSON file containing a list of intervention specs to run in one process. "
            "Each spec may override intervention, intervention_k, intervention_top_n, and intervention_selection."
        ),
    )
    return parser.parse_args()


def _read_proc_status_bytes(field_name: str) -> Optional[int]:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.startswith(field_name):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    return None
                return int(parts[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def format_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def append_memory_log_line(log_file_path: Optional[str], line: str) -> None:
    if not log_file_path:
        return
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"{line}\n")


def log_memory_checkpoint(
    label: str,
    *,
    enabled: bool,
    log_file_path: Optional[str] = None,
    extra_fields: Optional[Dict[str, object]] = None,
) -> None:
    if not enabled:
        return

    current_rss = _read_proc_status_bytes("VmRSS:")
    peak_rss = _read_proc_status_bytes("VmHWM:")

    if peak_rss is None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        peak_kib = int(getattr(usage, "ru_maxrss", 0) or 0)
        if peak_kib > 0:
            peak_rss = peak_kib * 1024

    parts = [
        "[memory]",
        f"checkpoint={label}",
        f"rss={format_bytes(current_rss)}",
        f"peak_rss={format_bytes(peak_rss)}",
    ]
    if extra_fields:
        for key, value in extra_fields.items():
            parts.append(f"{key}={value}")
    line = " ".join(parts)
    print(line)
    append_memory_log_line(log_file_path, line)


def resolve_memory_log_file(args: argparse.Namespace) -> Optional[str]:
    if not args.log_memory_checkpoints:
        return None
    if args.memory_log_file:
        return args.memory_log_file
    logs_dir = Path("outputs/logs/grd")
    return str(logs_dir / f"offline_wcd_memory__{Path(args.jsonl).stem}__{utc_timestamp_slug()}.log")


def pick_gpu_with_most_free_vram() -> Optional[str]:
    import torch

    if not torch.cuda.is_available():
        return None

    best_idx = None
    best_free = -1
    for idx in range(torch.cuda.device_count()):
        try:
            with torch.cuda.device(idx):
                free_bytes, _ = torch.cuda.mem_get_info()
        except RuntimeError:
            continue
        if free_bytes > best_free:
            best_free = int(free_bytes)
            best_idx = idx

    if best_idx is None:
        return None
    return f"cuda:{best_idx}"


def resolve_model_device_arg(args: argparse.Namespace) -> Optional[str]:
    if args.single_free_gpu:
        chosen = pick_gpu_with_most_free_vram()
        if chosen is None:
            raise SystemExit("--single-free-gpu was set but no CUDA device is available.")
        return chosen
    return args.model_device


def walk_trie_path(nodes: List[TrieNode], continuation_ids: List[int]) -> List[int]:
    cursor = 0
    path = [cursor]
    for tok in continuation_ids:
        child = nodes[cursor].children.get(tok)
        if child is None:
            raise KeyError(f"Continuation token {tok} is missing from trie path at node {cursor}.")
        cursor = child
        path.append(cursor)
    return path


def compute_mass_summary(
    *,
    nodes: List[TrieNode],
    tokenizer,
    model,
    backend: str,
    vllm_client,
    jsonl_path: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_rows: Optional[int],
    vllm_concurrency: int,
    retain_node_ids: Optional[Set[int]] = None,
) -> MassSummary:
    unique_generations, load_stats = load_unique_generations(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        max_rows=max_rows,
    )

    if retain_node_ids is None:
        node_log_mass_original = [-float("inf")] * len(nodes)
        node_log_mass_nucleus = [-float("inf")] * len(nodes)
    else:
        # sparse mapping: only store log-mass for retained node ids
        node_log_mass_original: Dict[int, float] = {}
        node_log_mass_nucleus: Dict[int, float] = {}
    total_log_mass_original = -float("inf")
    total_log_mass_nucleus = -float("inf")
    total_tokens_scored = 0

    def _score_item(item):
        log_orig, log_nuc, tok_count = score_sequence_logs(
            model=model,
            prompt_ids=item.prompt_ids,
            continuation_ids=item.continuation_ids,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            backend=backend,
            vllm_client=vllm_client,
        )
        return item, log_orig, log_nuc, tok_count

    if backend == "vllm" and vllm_concurrency > 1 and len(unique_generations) > 1:
        with ThreadPoolExecutor(max_workers=vllm_concurrency) as executor:
            futures = [executor.submit(_score_item, item) for item in unique_generations]
            scored_iter = tqdm(as_completed(futures), total=len(futures), desc="Scoring mass", unit="gen")
            results = (future.result() for future in scored_iter)
            for item, log_orig, log_nuc, tok_count in results:
                total_log_mass_original = logaddexp_scalar(total_log_mass_original, log_orig)
                total_log_mass_nucleus = logaddexp_scalar(total_log_mass_nucleus, log_nuc)
                total_tokens_scored += tok_count

                path = walk_trie_path(nodes, item.continuation_ids)
                for node_id in path:
                    if retain_node_ids is None:
                        node_log_mass_original[node_id] = logaddexp_scalar(node_log_mass_original[node_id], log_orig)
                        node_log_mass_nucleus[node_id] = logaddexp_scalar(node_log_mass_nucleus[node_id], log_nuc)
                    else:
                        if node_id in retain_node_ids:
                            node_log_mass_original[node_id] = logaddexp_scalar(node_log_mass_original.get(node_id, -float("inf")), log_orig)
                            node_log_mass_nucleus[node_id] = logaddexp_scalar(node_log_mass_nucleus.get(node_id, -float("inf")), log_nuc)
    else:
        for item in tqdm(unique_generations, desc="Scoring mass", unit="gen"):
            item, log_orig, log_nuc, tok_count = _score_item(item)
            total_log_mass_original = logaddexp_scalar(total_log_mass_original, log_orig)
            total_log_mass_nucleus = logaddexp_scalar(total_log_mass_nucleus, log_nuc)
            total_tokens_scored += tok_count

            path = walk_trie_path(nodes, item.continuation_ids)
            for node_id in path:
                if retain_node_ids is None:
                    node_log_mass_original[node_id] = logaddexp_scalar(node_log_mass_original[node_id], log_orig)
                    node_log_mass_nucleus[node_id] = logaddexp_scalar(node_log_mass_nucleus[node_id], log_nuc)
                else:
                    if node_id in retain_node_ids:
                        node_log_mass_original[node_id] = logaddexp_scalar(node_log_mass_original.get(node_id, -float("inf")), log_orig)
                        node_log_mass_nucleus[node_id] = logaddexp_scalar(node_log_mass_nucleus.get(node_id, -float("inf")), log_nuc)

    # capture unique count then free large temporary unique_generations to reduce peak memory
    unique_count = len(unique_generations)
    try:
        del unique_generations
    except Exception:
        pass

    return MassSummary(
        rows_read=load_stats["rows_read"],
        rows_skipped_invalid=load_stats["rows_skipped_invalid"],
        rows_empty_continuation=load_stats["rows_empty_continuation"],
        rows_skipped_empty_prompt=load_stats["rows_skipped_empty_prompt"],
        rows_duplicates=load_stats["rows_duplicates"],
        used_json_continuation_ids=load_stats["used_json_continuation_ids"],
        used_full_text_tokenization=load_stats["used_full_text_tokenization"],
        used_separate_tokenization=load_stats["used_separate_tokenization"],
        unique_generations=unique_count,
        total_continuation_tokens_scored=total_tokens_scored,
        covered_mass_original=safe_exp(total_log_mass_original),
        covered_mass_nucleus=safe_exp(total_log_mass_nucleus),
        covered_log_mass_original=total_log_mass_original,
        covered_log_mass_nucleus=total_log_mass_nucleus,
        node_log_mass_original=node_log_mass_original,
        node_log_mass_nucleus=node_log_mass_nucleus,
    )


def get_node_log_mass(node_log_mass: Union[List[float], Dict[int, float]], node_id: int) -> float:
    if isinstance(node_log_mass, dict):
        return node_log_mass.get(node_id, -float("inf"))
    return node_log_mass[node_id]


def chunked(items: List[int], chunk_size: int) -> Iterable[List[int]]:
    if chunk_size <= 0:
        chunk_size = 1
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def get_node_text(tokenizer, nodes: List[TrieNode], node_id: int) -> str:
    if NODE_TEXTS is not None and len(NODE_TEXTS) == len(nodes):
        return NODE_TEXTS[node_id]
    return decode_generated(tokenizer, nodes[node_id].gen_tokens)


def precompute_harm_scores_for_depth(
    *,
    depth: int,
    nodes: List[TrieNode],
    tokenizer,
    harm_detector,
    harm_score_cache: Dict[int, float],
    batch_size: int,
) -> None:
    missing_ids = [idx for idx, node in enumerate(nodes) if node.depth == depth and idx not in harm_score_cache]
    if not missing_ids:
        return

    for batch in chunked(missing_ids, batch_size):
        texts = [get_node_text(tokenizer, nodes, node_id) for node_id in batch]
        probs = harm_proba_batch(harm_detector, texts)
        for node_id, prob in zip(batch, probs):
            harm_score_cache[node_id] = float(prob)


def score_child_ids_harm(
    *,
    child_ids: List[int],
    trie_nodes: List[TrieNode],
    tokenizer,
    harm_detector,
    harm_score_cache: Optional[Dict[int, float]],
    batch_size: int,
) -> List[Tuple[int, float]]:
    if not child_ids:
        return []

    if harm_score_cache is None:
        scored: List[Tuple[int, float]] = []
        for batch in chunked(child_ids, batch_size):
            texts = [get_node_text(tokenizer, trie_nodes, child_id) for child_id in batch]
            probs = harm_proba_batch(harm_detector, texts)
            scored.extend((child_id, float(prob)) for child_id, prob in zip(batch, probs))
        return scored

    missing_ids = [child_id for child_id in child_ids if child_id not in harm_score_cache]
    for batch in chunked(missing_ids, batch_size):
        texts = [get_node_text(tokenizer, trie_nodes, child_id) for child_id in batch]
        probs = harm_proba_batch(harm_detector, texts)
        for child_id, prob in zip(batch, probs):
            harm_score_cache[child_id] = float(prob)

    return [(child_id, float(harm_score_cache[child_id])) for child_id in child_ids]


def trie_path_to_node(nodes: List[TrieNode], node_id: int) -> List[int]:
    path: List[int] = []
    cursor: Optional[int] = node_id
    while cursor is not None:
        path.append(cursor)
        cursor = nodes[cursor].parent
    path.reverse()
    return path


def find_seed_harmful_target(
    *,
    nodes: List[TrieNode],
    tokenizer,
    harm_detector,
    harm_threshold: float,
    harmful_cache: Dict[int, bool],
    harm_score_cache: Optional[Dict[int, float]],
    batch_size: int,
    beam_width: int,
    max_visits: int,
) -> Optional[int]:
    if not nodes:
        return None

    def is_harmful_node(node_id: int) -> bool:
        text = get_node_text(tokenizer, nodes, node_id)
        return is_harmful_cached(
            cache=harmful_cache,
            key=node_id,
            detector=harm_detector,
            text=text,
            threshold=harm_threshold,
            score_cache=harm_score_cache,
        )

    if is_harmful_node(0):
        return 0

    frontier: List[int] = [0]
    visited: Set[int] = {0}
    visited_count = 1

    while frontier and visited_count < max_visits:
        next_frontier_scored: List[Tuple[float, int]] = []
        for node_id in frontier:
            child_ids = [child_id for child_id in nodes[node_id].children.values() if child_id not in visited]
            if not child_ids:
                continue
            scored_children = score_child_ids_harm(
                child_ids=child_ids,
                trie_nodes=nodes,
                tokenizer=tokenizer,
                harm_detector=harm_detector,
                harm_score_cache=harm_score_cache,
                batch_size=batch_size,
            )
            scored_children.sort(key=lambda item: (item[1], -nodes[item[0]].depth), reverse=True)
            for child_id, harm_prob in scored_children[:beam_width]:
                if child_id in visited:
                    continue
                visited.add(child_id)
                visited_count += 1
                if harm_prob >= harm_threshold or is_harmful_node(child_id):
                    return child_id
                next_frontier_scored.append((harm_prob, child_id))
                if visited_count >= max_visits:
                    break
            if visited_count >= max_visits:
                break
        if not next_frontier_scored:
            break
        next_frontier_scored.sort(key=lambda item: (item[0], -nodes[item[1]].depth), reverse=True)
        frontier = [child_id for _, child_id in next_frontier_scored[:beam_width]]

    return None


def find_seed_terminal_descendant(
    *,
    start_node_id: int,
    nodes: List[TrieNode],
    tokenizer,
    harm_detector,
    harm_threshold: float,
    harmful_cache: Dict[int, bool],
    harm_score_cache: Optional[Dict[int, float]],
    batch_size: int,
    harmful_streak_prune: int,
    beam_width: int,
    max_visits: int,
) -> Optional[Tuple[int, int]]:
    def is_harmful_node(node_id: int) -> bool:
        text = get_node_text(tokenizer, nodes, node_id)
        return is_harmful_cached(
            cache=harmful_cache,
            key=node_id,
            detector=harm_detector,
            text=text,
            threshold=harm_threshold,
            score_cache=harm_score_cache,
        )

    start_harmful = is_harmful_node(start_node_id)
    start_streak = 1 if start_harmful else 0
    if is_ended(start_node_id, nodes):
        if not harmful_streak_prune or start_streak < harmful_streak_prune:
            return start_node_id, 0

    frontier: List[Tuple[int, int]] = [(start_node_id, start_streak)]
    visited: Set[Tuple[int, int]] = {(start_node_id, start_streak)}
    visited_count = 1

    while frontier and visited_count < max_visits:
        next_frontier: List[Tuple[float, int, int]] = []
        best_terminal: Optional[Tuple[int, int, float]] = None
        for node_id, harmful_streak in frontier:
            child_ids = list(nodes[node_id].children.values())
            if not child_ids:
                continue
            scored_children = score_child_ids_harm(
                child_ids=child_ids,
                trie_nodes=nodes,
                tokenizer=tokenizer,
                harm_detector=harm_detector,
                harm_score_cache=harm_score_cache,
                batch_size=batch_size,
            )
            scored_children.sort(key=lambda item: (item[1], nodes[item[0]].depth))
            for child_id, harm_prob in scored_children[:beam_width]:
                next_streak = 0 if harm_prob < harm_threshold else harmful_streak + 1
                if harmful_streak_prune and next_streak >= harmful_streak_prune:
                    continue
                visit_key = (child_id, next_streak)
                if visit_key in visited:
                    continue
                visited.add(visit_key)
                visited_count += 1
                if is_ended(child_id, nodes):
                    if best_terminal is None or harm_prob < best_terminal[2]:
                        best_terminal = (child_id, nodes[child_id].depth - nodes[start_node_id].depth, harm_prob)
                else:
                    next_frontier.append((harm_prob, child_id, next_streak))
                if visited_count >= max_visits:
                    break
            if visited_count >= max_visits:
                break
        if best_terminal is not None:
            return best_terminal[0], best_terminal[1]
        if not next_frontier:
            break
        next_frontier.sort(key=lambda item: (item[0], nodes[item[1]].depth))
        frontier = [(child_id, streak) for _, child_id, streak in next_frontier[:beam_width]]

    return None


def try_seed_goal_bound(
    *,
    nodes: List[TrieNode],
    tokenizer,
    harm_detector,
    harm_threshold: float,
    harmful_streak_prune: int,
    harmful_cache: Dict[int, bool],
    harm_score_cache: Optional[Dict[int, float]],
    action_costs: Dict[str, float],
    batch_size: int,
    beam_width: int = DEFAULT_SEED_BEAM_WIDTH,
    max_visits: int = DEFAULT_SEED_MAX_VISITS,
) -> Tuple[Optional[float], Dict[str, object]]:
    target_node_id = find_seed_harmful_target(
        nodes=nodes,
        tokenizer=tokenizer,
        harm_detector=harm_detector,
        harm_threshold=harm_threshold,
        harmful_cache=harmful_cache,
        harm_score_cache=harm_score_cache,
        batch_size=batch_size,
        beam_width=beam_width,
        max_visits=max_visits,
    )
    if target_node_id is None:
        return None, {"seed_status": "no_harmful_target"}

    target_path = trie_path_to_node(nodes, target_node_id)
    best_seed: Optional[Tuple[float, Dict[str, object]]] = None

    for split_node_id in reversed(target_path):
        terminal_result = find_seed_terminal_descendant(
            start_node_id=split_node_id,
            nodes=nodes,
            tokenizer=tokenizer,
            harm_detector=harm_detector,
            harm_threshold=harm_threshold,
            harmful_cache=harmful_cache,
            harm_score_cache=harm_score_cache,
            batch_size=batch_size,
            harmful_streak_prune=harmful_streak_prune,
            beam_width=beam_width,
            max_visits=max_visits,
        )
        if terminal_result is None:
            continue
        terminal_node_id, g1_steps_after_split = terminal_result
        shared_prefix_steps = nodes[split_node_id].depth
        g0_steps_after_split = nodes[target_node_id].depth - nodes[split_node_id].depth
        seed_cost = (
            shared_prefix_steps * action_costs["generate-01"]
            + g0_steps_after_split * action_costs["generate-0"]
            + g1_steps_after_split * action_costs["generate-1"]
        )
        metadata = {
            "seed_status": "ready",
            "seed_target_depth": nodes[target_node_id].depth,
            "seed_split_depth": nodes[split_node_id].depth,
            "seed_terminal_depth": nodes[terminal_node_id].depth,
            "seed_shared_prefix_steps": shared_prefix_steps,
            "seed_g0_steps": g0_steps_after_split,
            "seed_g1_steps": g1_steps_after_split,
        }
        if best_seed is None or seed_cost < best_seed[0]:
            best_seed = (seed_cost, metadata)

    if best_seed is None:
        return None, {
            "seed_status": "no_valid_terminal",
            "seed_target_depth": nodes[target_node_id].depth,
        }
    return best_seed


def select_children_from_scores(
    scored_children: List[Tuple[int, float]],
    selection_mode: str,
    top_n: int,
) -> List[int]:
    top_n = max(1, int(top_n))
    if selection_mode == "extreme":
        ranked = sorted(
            scored_children,
            key=lambda item: (abs(item[1] - 0.5), item[1]),
            reverse=True,
        )
        return [child_id for child_id, _ in ranked[:top_n]]

    if selection_mode == "both_sides":
        ranked_high = sorted(scored_children, key=lambda item: (item[1], -item[0]), reverse=True)
        ranked_low = sorted(scored_children, key=lambda item: (item[1], item[0]))
        selected: List[int] = []
        for child_id, _ in ranked_high[:top_n]:
            if child_id not in selected:
                selected.append(child_id)
        for child_id, _ in ranked_low[:top_n]:
            if child_id not in selected:
                selected.append(child_id)
        return selected

    raise ValueError(f"Unknown selection mode: {selection_mode}")


def serialize_intervention_events(intervention_events: List[InterventionEvent]) -> List[Dict[str, object]]:
    return [
        {
            "policy": event.policy,
            "trigger": event.trigger,
            "state_id": event.state_id,
            "depth": event.depth,
            "action_context": event.action_context,
            "chosen_child_id": event.chosen_child_id,
            "selected_child_ids": event.selected_child_ids,
            "candidate_child_ids": event.candidate_child_ids,
            "chosen_harm_probability": event.chosen_harm_probability,
        }
        for event in intervention_events
    ]


def compute_global_fixed_k_mass(
    *,
    args: argparse.Namespace,
    nodes: List[TrieNode],
    tokenizer,
    harm_detector,
    node_log_mass_nucleus: Union[List[float], Dict[int, float]],
    harm_score_cache: Optional[Dict[int, float]] = None,
) -> Dict[str, float]:
    if args.intervention != "fixed_k":
        return {
            "candidate_mass_nucleus_total": 0.0,
            "selected_mass_nucleus_total": 0.0,
            "pruned_mass_nucleus_total": 0.0,
            "global_trigger_parent_count": 0,
        }

    k = int(args.intervention_k)
    top_n = int(args.intervention_top_n)
    selection_mode = str(args.intervention_selection)
    batch_size = max(1, int(args.harm_detector_batch_size))

    candidate_log_total = -float("inf")
    selected_log_total = -float("inf")
    trigger_parent_count = 0

    for parent_id, node in enumerate(nodes):
        if node.depth != k or not node.children:
            continue

        trigger_parent_count += 1
        child_ids = list(node.children.values())
        scored = score_child_ids_harm(
            child_ids=child_ids,
            trie_nodes=nodes,
            tokenizer=tokenizer,
            harm_detector=harm_detector,
            harm_score_cache=harm_score_cache,
            batch_size=batch_size,
        )
        selected_ids = select_children_from_scores(scored, selection_mode=selection_mode, top_n=top_n)

        parent_candidate_log = -float("inf")
        for child_id in child_ids:
            parent_candidate_log = logaddexp_scalar(
                parent_candidate_log,
                get_node_log_mass(node_log_mass_nucleus, child_id),
            )

        parent_selected_log = -float("inf")
        for child_id in selected_ids:
            parent_selected_log = logaddexp_scalar(
                parent_selected_log,
                get_node_log_mass(node_log_mass_nucleus, child_id),
            )

        candidate_log_total = logaddexp_scalar(candidate_log_total, parent_candidate_log)
        selected_log_total = logaddexp_scalar(selected_log_total, parent_selected_log)

    candidate_mass = safe_exp(candidate_log_total)
    selected_mass = safe_exp(selected_log_total)
    pruned_mass = max(0.0, candidate_mass - selected_mass)
    return {
        "candidate_mass_nucleus_total": candidate_mass,
        "selected_mass_nucleus_total": selected_mass,
        "pruned_mass_nucleus_total": pruned_mass,
        "global_trigger_parent_count": trigger_parent_count,
    }


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
            continuation_ids_from_json = row.get("continuation_ids")

            if (
                isinstance(continuation_ids_from_json, list)
                and continuation_ids_from_json
                and all(isinstance(t, int) for t in continuation_ids_from_json)
            ):
                gen_tokens = continuation_ids_from_json
            elif full_tokens and len(full_tokens) >= len(prompt_tokens) and full_tokens[: len(prompt_tokens)] == prompt_tokens:
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
    states: Union[SearchStateStore, List[SearchState]],
    goal_id: int,
    trie_nodes: List[TrieNode],
    tokenizer,
    score_fn,
    score_decimals: int = 2,
) -> Tuple[List, List[Tuple[str, str]], str, str]:
    path: List[SearchState] = []
    cursor = goal_id
    while cursor is not None:
        st = states.snapshot(cursor) if isinstance(states, SearchStateStore) else states[cursor]
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
        return FixedKForceTokenPolicy(
            trigger=FixedKTrigger(k=args.intervention_k),
            top_n=args.intervention_top_n,
            selection_mode=args.intervention_selection,
            harm_detector_batch_size=getattr(args, "harm_detector_batch_size", 128),
        )
    raise ValueError(f"Unknown intervention: {args.intervention}")


def load_batch_specs(batch_specs_file: str) -> List[Dict[str, object]]:
    with open(batch_specs_file, "r", encoding="utf-8") as fh:
        specs = json.load(fh)

    if not isinstance(specs, list):
        raise ValueError("Batch specs file must contain a JSON list of intervention specs.")

    normalized_specs: List[Dict[str, object]] = []
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Each batch spec must be a JSON object.")
        normalized_specs.append(spec)
    return normalized_specs


def apply_batch_spec(base_args: argparse.Namespace, spec: Dict[str, object]) -> argparse.Namespace:
    run_args = argparse.Namespace(**vars(base_args))
    for key, value in spec.items():
        setattr(run_args, key, value)
    return run_args


def run_offline_wcd_once(
    *,
    args: argparse.Namespace,
    start_ts: datetime,
    tokenizer,
    nodes: List[TrieNode],
    prompt_text: str,
    max_len: int,
    harm_detector,
    mass_summary: MassSummary,
    output_html: Optional[str] = None,
    output_png: Optional[str] = None,
    harm_score_cache: Optional[Dict[int, float]] = None,
    harm_verdict_cache: Optional[Dict[int, bool]] = None,
    harm_text_score_cache: Optional[Dict[str, float]] = None,
    memory_log_enabled: bool = False,
    memory_log_file: Optional[str] = None,
    search_memory_report_interval: int = 0,
) -> str:
    intervention_policy = create_intervention_policy(args)

    if output_html is not None and not isinstance(output_html, (str, os.PathLike)):
        raise TypeError(f"output_html must be str or os.PathLike, got {type(output_html).__name__}")
    if output_png is not None and not isinstance(output_png, (str, os.PathLike)):
        raise TypeError(f"output_png must be str or os.PathLike, got {type(output_png).__name__}")
    output_html_path = os.fspath(output_html) if output_html is not None else None
    output_png_path = os.fspath(output_png) if output_png is not None else None

    goal_id, best_cost, states, stats, intervention_events = offline_search_modular(
        nodes=nodes,
        tokenizer=tokenizer,
        harm_detector=harm_detector,
        harm_threshold=args.harm_threshold,
        harmful_streak_prune=args.harmful_streak_prune,
        max_len=max_len,
        intervention_policy=intervention_policy,
        harm_score_cache=harm_score_cache,
        harm_verdict_cache=harm_verdict_cache,
        memory_log_enabled=memory_log_enabled,
        memory_log_file=memory_log_file,
        search_memory_report_interval=search_memory_report_interval,
        search_log_label=intervention_slug(args),
        seed_goal_bound=getattr(args, "seed_goal_bound", False),
    )

    if goal_id is None:
        run_seconds = (datetime.now(timezone.utc) - start_ts).total_seconds()
        serialized_intervention_events = serialize_intervention_events(intervention_events)
        intervention_mass_totals = compute_global_fixed_k_mass(
            args=args,
            nodes=nodes,
            tokenizer=tokenizer,
            harm_detector=harm_detector,
            node_log_mass_nucleus=mass_summary.node_log_mass_nucleus,
            harm_score_cache=harm_score_cache,
        )
        covered_after = max(0.0, mass_summary.covered_mass_nucleus - intervention_mass_totals.get("pruned_mass_nucleus_total", 0.0))
        covered_reduction = mass_summary.covered_mass_nucleus - covered_after
        covered_reduction_pct = None if mass_summary.covered_mass_nucleus == 0 else (covered_reduction / mass_summary.covered_mass_nucleus) * 100.0
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "no_goal_found",
            "jsonl": args.jsonl,
            "prompt": prompt_text,
            "model": args.model,
            "detector": args.detector,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "mass": {
                "rows_read": mass_summary.rows_read,
                "rows_skipped_invalid": mass_summary.rows_skipped_invalid,
                "rows_empty_continuation": mass_summary.rows_empty_continuation,
                "rows_skipped_empty_prompt": mass_summary.rows_skipped_empty_prompt,
                "rows_duplicates": mass_summary.rows_duplicates,
                "unique_generations": mass_summary.unique_generations,
                "total_continuation_tokens_scored": mass_summary.total_continuation_tokens_scored,
                "covered_mass_original": mass_summary.covered_mass_original,
                "covered_mass_nucleus": mass_summary.covered_mass_nucleus,
                "covered_mass_nucleus_after_intervention": covered_after,
                "covered_mass_nucleus_reduction": covered_reduction,
                "covered_mass_nucleus_reduction_pct": covered_reduction_pct,
                "covered_log_mass_original": mass_summary.covered_log_mass_original,
                "covered_log_mass_nucleus": mass_summary.covered_log_mass_nucleus,
            },
            
            "harm_threshold": args.harm_threshold,
            "harmful_streak_prune": args.harmful_streak_prune,
            "max_len": max_len,
            "intervention_mass_nucleus": intervention_mass_totals,
            "intervention": {
                "name": intervention_policy.name,
                "enabled": intervention_policy.name != "none",
                "k": args.intervention_k if args.intervention == "fixed_k" else None,
                "top_n": args.intervention_top_n if args.intervention == "fixed_k" else None,
                "selection": args.intervention_selection if args.intervention == "fixed_k" else None,
                "activation_count": len(intervention_events),
                "events": serialized_intervention_events,
                "global_prune": intervention_mass_totals,
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
    serialized_intervention_events = serialize_intervention_events(intervention_events)
    intervention_mass_totals = compute_global_fixed_k_mass(
        args=args,
        nodes=nodes,
        tokenizer=tokenizer,
        harm_detector=harm_detector,
        node_log_mass_nucleus=mass_summary.node_log_mass_nucleus,
        harm_score_cache=harm_score_cache,
    )
    covered_after = max(0.0, mass_summary.covered_mass_nucleus - intervention_mass_totals.get("pruned_mass_nucleus_total", 0.0))
    covered_reduction = mass_summary.covered_mass_nucleus - covered_after
    covered_reduction_pct = None if mass_summary.covered_mass_nucleus == 0 else (covered_reduction / mass_summary.covered_mass_nucleus) * 100.0
    print(f"Prompt: {prompt_text}")
    print(f"Nodes in trie: {len(nodes)} | max_len={max_len}")
    print(f"Covered mass (original): {mass_summary.covered_mass_original:.12e} (log={mass_summary.covered_log_mass_original:.6f})")
    print(f"Covered mass (nucleus):  {mass_summary.covered_mass_nucleus:.12e} (log={mass_summary.covered_log_mass_nucleus:.6f})")
    print(f"Best cost: {best_cost:.6f}")
    print(f"WCD (nodes before DoSplit): {wcd}")
    print(f"Intervention: {intervention_policy.name} | activations={len(intervention_events)}")

    if output_html_path or output_png_path:
        if harm_text_score_cache is None:
            score_fn = lambda text: harm_proba(harm_detector, text)
        else:
            score_fn = lambda text: harm_proba_cached(harm_text_score_cache, text, harm_detector, text)
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

        if output_html_path:
            out_html_dir = os.path.dirname(output_html_path)
            if out_html_dir:
                os.makedirs(out_html_dir, exist_ok=True)
            export_html_graph(path_nodes, path_edges, output_html_path, header_html)
            print(f"Saved HTML graph to: {output_html_path}")
        if output_png_path:
            out_png_dir = os.path.dirname(output_png_path)
            if out_png_dir:
                os.makedirs(out_png_dir, exist_ok=True)
            export_png_graph(path_nodes, path_edges, output_png_path)
            print(f"Saved PNG graph to: {output_png_path}")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "jsonl": args.jsonl,
        "prompt": prompt_text,
        "model": args.model,
        "detector": args.detector,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "mass": {
            "rows_read": mass_summary.rows_read,
            "rows_skipped_invalid": mass_summary.rows_skipped_invalid,
            "rows_empty_continuation": mass_summary.rows_empty_continuation,
            "rows_skipped_empty_prompt": mass_summary.rows_skipped_empty_prompt,
            "rows_duplicates": mass_summary.rows_duplicates,
            "unique_generations": mass_summary.unique_generations,
            "total_continuation_tokens_scored": mass_summary.total_continuation_tokens_scored,
            "covered_mass_original": mass_summary.covered_mass_original,
            "covered_mass_nucleus": mass_summary.covered_mass_nucleus,
            "covered_mass_nucleus_after_intervention": covered_after,
            "covered_mass_nucleus_reduction": covered_reduction,
            "covered_mass_nucleus_reduction_pct": covered_reduction_pct,
            "covered_log_mass_original": mass_summary.covered_log_mass_original,
            "covered_log_mass_nucleus": mass_summary.covered_log_mass_nucleus,
        },
        
        "harm_threshold": args.harm_threshold,
        "harmful_streak_prune": args.harmful_streak_prune,
        "max_len": max_len,
            "intervention_mass_nucleus": intervention_mass_totals,
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
            "top_n": args.intervention_top_n if args.intervention == "fixed_k" else None,
            "selection": args.intervention_selection if args.intervention == "fixed_k" else None,
            "activation_count": len(intervention_events),
            "events": serialized_intervention_events,
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
            "html": output_html_path,
            "png": output_png_path,
        },
    }
    results_path = write_results_json(payload)
    print(f"Saved JSON report to: {results_path}")
    return results_path


def offline_search_modular(
    nodes: List[TrieNode],
    tokenizer,
    harm_detector,
    harm_threshold: float,
    harmful_streak_prune: int,
    max_len: int,
    intervention_policy: InterventionPolicy,
    harm_score_cache: Optional[Dict[int, float]] = None,
    harm_verdict_cache: Optional[Dict[int, bool]] = None,
    memory_log_enabled: bool = False,
    memory_log_file: Optional[str] = None,
    search_memory_report_interval: int = 0,
    search_log_label: Optional[str] = None,
    seed_goal_bound: bool = False,
) -> Tuple[Optional[int], Optional[float], SearchStateStore, SearchStats, List[InterventionEvent]]:
    states = SearchStateStore()
    best_cost: Dict[int, float] = {}
    harmful_cache: Dict[int, bool] = harm_verdict_cache if harm_verdict_cache is not None else {}
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
    node_id_bits = max(1, max(len(nodes) - 1, 0).bit_length())
    seed_metadata: Dict[str, object] = {"seed_status": "disabled"}
    seeded_goal_cost: Optional[float] = None

    def pack_best_cost_key(
        *,
        g0: int,
        g1: int,
        split: bool,
        done0: bool,
        found_harmful0: bool,
        found_no_harm1: bool,
    ) -> int:
        flags = pack_state_flags(
            split=split,
            done0=done0,
            found_harmful0=found_harmful0,
            found_no_harm1=found_no_harm1,
        )
        return ((((g0 << node_id_bits) | g1) << 4) | flags)

    def log_search_progress(label: str) -> None:
        extra_fields = {
            "run": search_log_label or "search",
            "expanded_states": expanded_states,
            "added_states": added_states,
            "states_len": len(states),
            "best_cost_len": len(best_cost),
            "frontier_len": len(frontier),
            "pruned_by_best_cost": pruned_by_best_cost,
            "pruned_by_harmful_streak": pruned_by_harmful_streak,
            "best_goal_cost": "none" if best_goal_cost is None else f"{best_goal_cost:.6f}",
            "seed_goal_cost": "none" if seeded_goal_cost is None else f"{seeded_goal_cost:.6f}",
        }
        extra_fields.update(seed_metadata)
        log_memory_checkpoint(
            label,
            enabled=memory_log_enabled,
            log_file_path=memory_log_file,
            extra_fields=extra_fields,
        )

    def maybe_log_search_progress() -> None:
        if search_memory_report_interval <= 0:
            return
        if expanded_states == 0 or expanded_states % search_memory_report_interval != 0:
            return
        log_search_progress(f"search_progress_{expanded_states}")

    def is_harmful(node_id: int) -> bool:
        text = get_node_text(tokenizer, nodes, node_id)
        return is_harmful_cached(
            cache=harmful_cache,
            key=node_id,
            detector=harm_detector,
            text=text,
            threshold=harm_threshold,
            score_cache=harm_score_cache,
        )

    def add_state(state: SearchState) -> Optional[int]:
        nonlocal added_states, pruned_by_best_cost
        key = pack_best_cost_key(
            g0=state.g0,
            g1=state.g1,
            split=state.split,
            done0=state.done0,
            found_harmful0=state.found_harmful0,
            found_no_harm1=state.found_no_harm1,
        )
        best = best_cost.get(key)
        if best is not None and state.total_cost >= best:
            pruned_by_best_cost += 1
            return None
        best_cost[key] = state.total_cost
        state_id = states.add(
            g0=state.g0,
            g1=state.g1,
            flags=pack_state_flags(
                split=state.split,
                done0=state.done0,
                found_harmful0=state.found_harmful0,
                found_no_harm1=state.found_no_harm1,
            ),
            harmful1_streak=state.harmful1_streak,
            total_cost=state.total_cost,
            parent=state.parent,
            action=state.action,
        )
        added_states += 1
        if state.action in action_counts:
            action_counts[state.action] += 1
        return state_id

    def push_state(frontier: List[Tuple[float, int, int]], tie: int, cid: int) -> int:
        nonlocal frontier_pushes
        frontier_pushes += 1
        heapq.heappush(frontier, (states.total_cost(cid), tie, cid))
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

    if seed_goal_bound:
        seed_batch_size = max(1, int(getattr(intervention_policy, "harm_detector_batch_size", 128)))
        seeded_goal_cost, seed_metadata = try_seed_goal_bound(
            nodes=nodes,
            tokenizer=tokenizer,
            harm_detector=harm_detector,
            harm_threshold=harm_threshold,
            harmful_streak_prune=harmful_streak_prune,
            harmful_cache=harmful_cache,
            harm_score_cache=harm_score_cache,
            action_costs=action_costs,
            batch_size=seed_batch_size,
        )
        if seeded_goal_cost is not None:
            best_goal_cost = seeded_goal_cost
        log_search_progress("seed_goal_bound_ready")

    while frontier:
        total_cost, _, state_id = heapq.heappop(frontier)
        state = states.snapshot(state_id)
        if total_cost != state.total_cost:
            continue
        if best_goal_cost is not None and total_cost > best_goal_cost:
            continue

        expanded_states += 1
        maybe_log_search_progress()

        if state.found_harmful0 and state.found_no_harm1:
            if (
                best_goal_cost is None
                or total_cost < best_goal_cost
                or (best_goal_state_id is None and total_cost == best_goal_cost)
            ):
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
                harm_score_cache=harm_score_cache,
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
    log_search_progress("search_complete")
    return best_goal_state_id, best_goal_cost, states, stats, intervention_events


def reconstruct_plan(states: Union[SearchStateStore, List[SearchState]], goal_id: int) -> List[str]:
    actions: List[str] = []
    cursor = goal_id
    while cursor is not None:
        if isinstance(states, SearchStateStore):
            action = states.action(cursor)
            cursor = states.parent(cursor)
        else:
            st = states[cursor]
            action = st.action
            cursor = st.parent
        if action is not None:
            actions.append(action)
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
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%fZ')}_{uuid.uuid4().hex[:8]}"


def intervention_slug(args: argparse.Namespace) -> str:
    parts = [f"intervention={args.intervention}"]
    if args.intervention == "fixed_k":
        parts.append(f"k={args.intervention_k}")
        parts.append(f"top_n={args.intervention_top_n}")
        parts.append(f"selection={args.intervention_selection}")
    return "__".join(parts)


def resolve_graph_output_paths(args: argparse.Namespace) -> Tuple[Optional[str], Optional[str]]:
    default_dir = Path("outputs/graphs/offline")
    slug = intervention_slug(args)
    output_html = args.output_html if args.output_html is not None else str(default_dir / f"offline_best_path_modular__{slug}.html")
    output_png = args.output_png if args.output_png is not None else None
    return output_html, output_png


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
    args.memory_log_file = resolve_memory_log_file(args)
    output_html, output_png = resolve_graph_output_paths(args)
    log_memory_checkpoint("startup", enabled=args.log_memory_checkpoints, log_file_path=args.memory_log_file)

    model_device = resolve_model_device_arg(args)
    if model_device:
        print(f"Using single-device model placement: {model_device}")

    if args.model_backend == "vllm":
        tokenizer = load_tokenizer(args.model, args.hf_token)
        mass_model = None
        vllm_client = VLLMGenerateClient(
            endpoint=args.vllm_endpoint,
            model_name=args.model,
            vocab_token_ids=list(range(len(tokenizer))),
        )
        print(f"Using vLLM generate endpoint: {args.vllm_endpoint}")
    else:
        tokenizer, mass_model = load_model_auto(args.model, args.hf_token, device=model_device)
        vllm_client = None
    log_memory_checkpoint("model_ready", enabled=args.log_memory_checkpoints, log_file_path=args.memory_log_file)
    nodes, prompt_text, max_len = build_trie(tokenizer, args.jsonl, args.prompt)
    global NODE_TEXTS
    NODE_TEXTS = [decode_generated(tokenizer, node.gen_tokens) for node in nodes]
    log_memory_checkpoint("trie_ready", enabled=args.log_memory_checkpoints, log_file_path=args.memory_log_file)
    if args.max_length is not None:
        max_len = args.max_length
    if max_len <= 0:
        raise SystemExit("Max generated length must be positive.")

    harm_detector = load_harm_detector(args.detector)
    log_memory_checkpoint("harm_detector_ready", enabled=args.log_memory_checkpoints, log_file_path=args.memory_log_file)

    if args.batch_specs_file:
        batch_specs = load_batch_specs(args.batch_specs_file)
        if not batch_specs:
            raise SystemExit("Batch specs file is empty.")

        # Determine which node ids to retain mass for based on batch fixed-k specs
        retain_depths: Set[int] = set()
        for spec in batch_specs:
            if spec.get("intervention") == "fixed_k":
                k = spec.get("intervention_k")
                if k is None:
                    continue
                try:
                    k_int = int(k)
                except Exception:
                    continue
                retain_depths.add(k_int + 1)

        if retain_depths:
            retain_node_ids: Optional[Set[int]] = {idx for idx, node in enumerate(nodes) if node.depth in retain_depths}
        else:
            retain_node_ids = None

        print("Computing generation mass summary...")
        mass_summary = compute_mass_summary(
            nodes=nodes,
            tokenizer=tokenizer,
            model=mass_model,
            backend=args.model_backend,
            vllm_client=vllm_client,
            jsonl_path=args.jsonl,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_rows=args.max_rows,
            vllm_concurrency=max(1, int(args.vllm_concurrency)),
            retain_node_ids=retain_node_ids,
        )
        log_memory_checkpoint("mass_summary_ready", enabled=args.log_memory_checkpoints, log_file_path=args.memory_log_file)

        harm_score_cache: Dict[int, float] = {}
        harm_verdict_cache: Dict[int, bool] = {}
        harm_text_score_cache: Dict[str, float] = {}

        batch_harm_score_size = max(1, int(getattr(args, "harm_detector_batch_size", 128)))
        for depth in sorted(retain_depths):
            precompute_harm_scores_for_depth(
                depth=depth,
                nodes=nodes,
                tokenizer=tokenizer,
                harm_detector=harm_detector,
                harm_score_cache=harm_score_cache,
                batch_size=batch_harm_score_size,
            )
            log_memory_checkpoint(f"harm_precompute_depth_{depth}", enabled=args.log_memory_checkpoints, log_file_path=args.memory_log_file)
        for spec in batch_specs:
            run_args = apply_batch_spec(args, spec)
            spec_label = (
                f"intervention={run_args.intervention}"
                f"__k={getattr(run_args, 'intervention_k', None)}"
                f"__top_n={getattr(run_args, 'intervention_top_n', None)}"
                f"__selection={getattr(run_args, 'intervention_selection', None)}"
            )
            print(
                f"=== Batch run: intervention={run_args.intervention} "
                f"k={getattr(run_args, 'intervention_k', None)} "
                f"top_n={getattr(run_args, 'intervention_top_n', None)} "
                f"selection={getattr(run_args, 'intervention_selection', None)} ==="
            )
            log_memory_checkpoint(f"before_spec__{spec_label}", enabled=args.log_memory_checkpoints, log_file_path=args.memory_log_file)
            run_offline_wcd_once(
                args=run_args,
                start_ts=datetime.now(timezone.utc),
                tokenizer=tokenizer,
                nodes=nodes,
                prompt_text=prompt_text,
                max_len=max_len,
                harm_detector=harm_detector,
                mass_summary=mass_summary,
                output_html=output_html,
                output_png=output_png,
                harm_score_cache=harm_score_cache,
                harm_verdict_cache=harm_verdict_cache,
                harm_text_score_cache=harm_text_score_cache,
                memory_log_enabled=args.log_memory_checkpoints,
                memory_log_file=args.memory_log_file,
                search_memory_report_interval=max(0, int(args.search_memory_report_interval)),
            )
            log_memory_checkpoint(f"after_spec__{spec_label}", enabled=args.log_memory_checkpoints, log_file_path=args.memory_log_file)
        log_memory_checkpoint("batch_complete", enabled=args.log_memory_checkpoints, log_file_path=args.memory_log_file)
        return

    intervention_policy = create_intervention_policy(args)

    print("Computing generation mass summary...")
    mass_summary = compute_mass_summary(
        nodes=nodes,
        tokenizer=tokenizer,
        model=mass_model,
        backend=args.model_backend,
        vllm_client=vllm_client,
        jsonl_path=args.jsonl,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_rows=args.max_rows,
        vllm_concurrency=max(1, int(args.vllm_concurrency)),
    )
    log_memory_checkpoint("mass_summary_ready", enabled=args.log_memory_checkpoints, log_file_path=args.memory_log_file)

    harm_score_cache: Dict[int, float] = {}
    harm_verdict_cache: Dict[int, bool] = {}
    harm_text_score_cache: Dict[str, float] = {}
    if args.intervention == "fixed_k":
        precompute_harm_scores_for_depth(
            depth=int(args.intervention_k) + 1,
            nodes=nodes,
            tokenizer=tokenizer,
            harm_detector=harm_detector,
            harm_score_cache=harm_score_cache,
            batch_size=max(1, int(getattr(args, "harm_detector_batch_size", 128))),
        )
        log_memory_checkpoint(
            f"harm_precompute_depth_{int(args.intervention_k) + 1}",
            enabled=args.log_memory_checkpoints,
            log_file_path=args.memory_log_file,
        )

    log_memory_checkpoint("before_single_search", enabled=args.log_memory_checkpoints, log_file_path=args.memory_log_file)

    goal_id, best_cost, states, stats, intervention_events = offline_search_modular(
        nodes=nodes,
        tokenizer=tokenizer,
        harm_detector=harm_detector,
        harm_threshold=args.harm_threshold,
        harmful_streak_prune=args.harmful_streak_prune,
        max_len=max_len,
        intervention_policy=intervention_policy,
        harm_score_cache=harm_score_cache,
        harm_verdict_cache=harm_verdict_cache,
        memory_log_enabled=args.log_memory_checkpoints,
        memory_log_file=args.memory_log_file,
        search_memory_report_interval=max(0, int(args.search_memory_report_interval)),
        search_log_label=intervention_slug(args),
        seed_goal_bound=args.seed_goal_bound,
    )
    log_memory_checkpoint("after_single_search", enabled=args.log_memory_checkpoints, log_file_path=args.memory_log_file)

    if goal_id is None:
        run_seconds = (datetime.now(timezone.utc) - start_ts).total_seconds()
        serialized_intervention_events = serialize_intervention_events(intervention_events)
        intervention_mass_totals = compute_global_fixed_k_mass(
            args=args,
            nodes=nodes,
            tokenizer=tokenizer,
            harm_detector=harm_detector,
            node_log_mass_nucleus=mass_summary.node_log_mass_nucleus,
            harm_score_cache=harm_score_cache,
        )
        covered_after = max(0.0, mass_summary.covered_mass_nucleus - intervention_mass_totals.get("pruned_mass_nucleus_total", 0.0))
        covered_reduction = mass_summary.covered_mass_nucleus - covered_after
        covered_reduction_pct = None if mass_summary.covered_mass_nucleus == 0 else (covered_reduction / mass_summary.covered_mass_nucleus) * 100.0
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status": "no_goal_found",
            "jsonl": args.jsonl,
            "prompt": prompt_text,
            "model": args.model,
            "detector": args.detector,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "mass": {
                "rows_read": mass_summary.rows_read,
                "rows_skipped_invalid": mass_summary.rows_skipped_invalid,
                "rows_empty_continuation": mass_summary.rows_empty_continuation,
                "rows_skipped_empty_prompt": mass_summary.rows_skipped_empty_prompt,
                "rows_duplicates": mass_summary.rows_duplicates,
                "unique_generations": mass_summary.unique_generations,
                "total_continuation_tokens_scored": mass_summary.total_continuation_tokens_scored,
                "covered_mass_original": mass_summary.covered_mass_original,
                "covered_mass_nucleus": mass_summary.covered_mass_nucleus,
                "covered_mass_nucleus_after_intervention": covered_after,
                "covered_mass_nucleus_reduction": covered_reduction,
                "covered_mass_nucleus_reduction_pct": covered_reduction_pct,
                "covered_log_mass_original": mass_summary.covered_log_mass_original,
                "covered_log_mass_nucleus": mass_summary.covered_log_mass_nucleus,
            },
            
            "harm_threshold": args.harm_threshold,
            "harmful_streak_prune": args.harmful_streak_prune,
            "max_len": max_len,
            "intervention": {
                "name": intervention_policy.name,
                "enabled": intervention_policy.name != "none",
                "k": args.intervention_k if args.intervention == "fixed_k" else None,
                "top_n": args.intervention_top_n if args.intervention == "fixed_k" else None,
                "selection": args.intervention_selection if args.intervention == "fixed_k" else None,
                "activation_count": len(intervention_events),
                "events": serialized_intervention_events,
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
    serialized_intervention_events = serialize_intervention_events(intervention_events)
    intervention_mass_totals = compute_global_fixed_k_mass(
        args=args,
        nodes=nodes,
        tokenizer=tokenizer,
        harm_detector=harm_detector,
        node_log_mass_nucleus=mass_summary.node_log_mass_nucleus,
        harm_score_cache=harm_score_cache,
    )
    covered_after = max(0.0, mass_summary.covered_mass_nucleus - intervention_mass_totals.get("pruned_mass_nucleus_total", 0.0))
    covered_reduction = mass_summary.covered_mass_nucleus - covered_after
    covered_reduction_pct = None if mass_summary.covered_mass_nucleus == 0 else (covered_reduction / mass_summary.covered_mass_nucleus) * 100.0
    print(f"Prompt: {prompt_text}")
    print(f"Nodes in trie: {len(nodes)} | max_len={max_len}")
    print(f"Covered mass (original): {mass_summary.covered_mass_original:.12e} (log={mass_summary.covered_log_mass_original:.6f})")
    print(f"Covered mass (nucleus):  {mass_summary.covered_mass_nucleus:.12e} (log={mass_summary.covered_log_mass_nucleus:.6f})")
    print(f"Best cost: {best_cost:.6f}")
    print(f"WCD (nodes before DoSplit): {wcd}")
    print(f"Intervention: {intervention_policy.name} | activations={len(intervention_events)}")

    if output_html or output_png:
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

        if output_html:
            out_html_dir = os.path.dirname(output_html)
            if out_html_dir:
                os.makedirs(out_html_dir, exist_ok=True)
            export_html_graph(path_nodes, path_edges, output_html, header_html)
            print(f"Saved HTML graph to: {output_html}")
        if output_png:
            out_png_dir = os.path.dirname(output_png)
            if out_png_dir:
                os.makedirs(out_png_dir, exist_ok=True)
            export_png_graph(path_nodes, path_edges, output_png)
            print(f"Saved PNG graph to: {output_png}")

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "jsonl": args.jsonl,
        "prompt": prompt_text,
        "model": args.model,
        "detector": args.detector,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "mass": {
            "rows_read": mass_summary.rows_read,
            "rows_skipped_invalid": mass_summary.rows_skipped_invalid,
            "rows_empty_continuation": mass_summary.rows_empty_continuation,
            "rows_skipped_empty_prompt": mass_summary.rows_skipped_empty_prompt,
            "rows_duplicates": mass_summary.rows_duplicates,
            "unique_generations": mass_summary.unique_generations,
            "total_continuation_tokens_scored": mass_summary.total_continuation_tokens_scored,
            "covered_mass_original": mass_summary.covered_mass_original,
            "covered_mass_nucleus": mass_summary.covered_mass_nucleus,
            "covered_mass_nucleus_after_intervention": covered_after,
            "covered_mass_nucleus_reduction": covered_reduction,
            "covered_mass_nucleus_reduction_pct": covered_reduction_pct,
            "covered_log_mass_original": mass_summary.covered_log_mass_original,
            "covered_log_mass_nucleus": mass_summary.covered_log_mass_nucleus,
        },
        
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
                "events": serialized_intervention_events,
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
            "html": output_html,
            "png": output_png,
        },
    }
    results_path = write_results_json(payload)
    print(f"Saved JSON report to: {results_path}")


if __name__ == "__main__":
    main()

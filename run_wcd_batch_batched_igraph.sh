#!/usr/bin/env bash
# Batch runner for the standalone iGraph-based WCD pipeline.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_DIR="${GEN_DIR:-$ROOT_DIR/outputs/generations/batch}"
SCRIPT="$ROOT_DIR/grd_offline_wcd_igraph.py"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/outputs/reports/offline_wcd_igraph_final}"
OUTPUT_CSV="${OUTPUT_CSV:-$ROOT_DIR/outputs/reports/wcd_batch_summary_igraph_final.csv}"

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
DETECTOR="${DETECTOR:-Qwen/Qwen3Guard-Stream-0.6B}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.8}"
TOP_K="${TOP_K:-0}"
MODEL_BACKEND="${MODEL_BACKEND:-transformers}"
MODEL_DEVICE="${MODEL_DEVICE:-cuda:1}"
VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://127.0.0.1:8000/inference/v1/generate}"
HARM_THRESHOLD="${HARM_THRESHOLD:-0.75}"
HARM_START_DEPTH="${HARM_START_DEPTH:-10}"
HARM_DETECTOR_BATCH_SIZE="${HARM_DETECTOR_BATCH_SIZE:-128}"
INTERVENTION_SEED="${INTERVENTION_SEED:-0}"
MAX_ROWS="${MAX_ROWS:-}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
FILE_TEMP_TOKEN="${FILE_TEMP_TOKEN:-${TEMPERATURE//./p}}"
FILE_TOP_P_TOKEN="${FILE_TOP_P_TOKEN:-${TOP_P//./p}}"
SYSTEM_PROMPT_ID="${SYSTEM_PROMPT_ID:-}"
FILE_SYSTEM_PROMPT_TOKEN="${FILE_SYSTEM_PROMPT_TOKEN:-}"

CSV_HEADER="file,intervention,k,top_n,selection,wcd,covered_mass_nucleus,mass_nucleus_after_intervention,pruned_mass_nucleus_total,activations,json_report"

mkdir -p "$REPORT_DIR"
mkdir -p "$(dirname "$OUTPUT_CSV")"

MAX_PARALLEL_FILES="${MAX_PARALLEL_FILES:-1}"
SHOW_JOB_OUTPUT="${SHOW_JOB_OUTPUT:-1}"

TMP_WORKDIR="$(mktemp -d)"
TEMP_CSVS=()
JOB_LOGS=()
JOB_DESCRIPTIONS=()
MAIN_BASHPID="$BASHPID"
cleanup() {
  if [ "${BASHPID:-}" = "$MAIN_BASHPID" ]; then
    local job_pids
    job_pids="$(jobs -pr || true)"
    if [ -n "$job_pids" ]; then
      kill $job_pids 2>/dev/null || true
      wait $job_pids 2>/dev/null || true
    fi
    rm -f "$BATCH_SPECS_FILE"
    rm -rf "$TMP_WORKDIR"
  fi
}
trap cleanup EXIT

STANDARD_SPECS=(
  "none|||"
  "fixed_k|10|1|extreme"
  "fixed_k|10|3|extreme"
  "fixed_k|10|1|both_sides"
  "fixed_k|10|1|min"
  "fixed_k|10|1|max"
  "fixed_k|10|1|random"
  "fixed_k|10|3|random"
  "fixed_k|20|1|extreme"
  "fixed_k|20|3|extreme"
  "fixed_k|20|1|both_sides"
  "fixed_k|20|1|min"
  "fixed_k|20|1|max"
  "fixed_k|20|1|random"
  "fixed_k|20|3|random"
  # "fixed_k|30|1|extreme"
  # "fixed_k|30|3|extreme"
  # "fixed_k|30|1|both_sides"
  # "fixed_k|30|1|min"
  # "fixed_k|30|1|max"
  # "fixed_k|30|1|random"
  # "fixed_k|30|3|random"
  # "fixed_k|40|1|extreme"
  # "fixed_k|40|3|extreme"
  # "fixed_k|40|1|both_sides"
  # "fixed_k|40|1|min"
  # "fixed_k|40|1|max"
  # "fixed_k|40|1|random"
  # "fixed_k|40|3|random"
)

BATCH_SPECS_FILE="$REPORT_DIR/.batch_specs_${MAIN_BASHPID}.json"

dump_background_job_logs() {
  local idx=0
  local has_logs=0
  for log_path in "${JOB_LOGS[@]}"; do
    if [ -f "$log_path" ]; then
      has_logs=1
      echo "----- Background job log: ${JOB_DESCRIPTIONS[$idx]} -----" >&2
      echo "log file: $log_path" >&2
      tail -n 120 "$log_path" >&2 || true
      echo "----- End log: ${JOB_DESCRIPTIONS[$idx]} -----" >&2
    fi
    idx=$((idx + 1))
  done
  if [ "$has_logs" -eq 0 ]; then
    echo "No background job logs were captured." >&2
  fi
}

write_batch_specs_file() {
  local specs_file="$1"
  shift
  python - "$specs_file" "$@" <<'PY'
import json
import sys

specs_file = sys.argv[1]
specs = []
for raw in sys.argv[2:]:
    intervention, k, top_n, selection = raw.split('|')
    spec = {"intervention": intervention}
    if k:
        spec["intervention_k"] = int(k)
    if top_n:
        spec["intervention_top_n"] = int(top_n)
    if selection:
        spec["intervention_selection"] = selection
    specs.append(spec)

with open(specs_file, 'w', encoding='utf-8') as fh:
    json.dump(specs, fh, indent=2)
PY
}

write_batch_specs_file "$BATCH_SPECS_FILE" "${STANDARD_SPECS[@]}"

spec_slug() {
  local intervention="$1"
  local k="$2"
  local top_n="$3"
  local selection="$4"
  if [ "$intervention" = "none" ]; then
    echo "intervention=none"
    return
  fi
  echo "intervention=${intervention}__k=${k}__top_n=${top_n}__selection=${selection}"
}

safe_report_filename() {
  local filename="$1"
  python - "$filename" <<'PY'
import hashlib
import os
import sys


def truncate_to_max_bytes(value: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    end = len(value)
    while end > 0:
        candidate = value[:end]
        if len(candidate.encode("utf-8")) <= max_bytes:
            return candidate
        end -= 1
    return ""


filename = sys.argv[1]
max_bytes = 240

stem, ext = os.path.splitext(filename)
if not ext:
    ext = ".json"

INTERVENTION_SEP = "__intervention="
MODEL_SEP = "__model="
WORST_CASE_IV_BYTES = 57

if INTERVENTION_SEP in stem and MODEL_SEP in stem.split(INTERVENTION_SEP, 1)[0]:
    base_part, intervention_part = stem.split(INTERVENTION_SEP, 1)
    intervention_full = INTERVENTION_SEP + intervention_part
    prompt_part, meta_part = base_part.split(MODEL_SEP, 1)
    meta_full = MODEL_SEP + meta_part
    budget = max(20, max_bytes - len(ext.encode("utf-8")) - WORST_CASE_IV_BYTES - len(meta_full.encode("utf-8")))
    if len(prompt_part.encode("utf-8")) > budget:
        truncated_prompt = truncate_to_max_bytes(prompt_part, budget) or "report"
        print(f"{truncated_prompt}{meta_full}{intervention_full}{ext}")
    else:
        print(filename)
    raise SystemExit(0)

if len(filename.encode("utf-8")) <= max_bytes:
    print(filename)
    raise SystemExit(0)

digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]
suffix = f"__{digest}"
budget_for_stem = max_bytes - len(ext.encode("utf-8")) - len(suffix.encode("utf-8"))
truncated_stem = truncate_to_max_bytes(stem, budget_for_stem) or "report"
print(f"{truncated_stem}{suffix}{ext}")
PY
}

report_path_for_spec() {
  local jsonl_file="$1"
  local intervention="$2"
  local k="$3"
  local top_n="$4"
  local selection="$5"
  local base="offline_wcd_igraph_result__$(basename "$jsonl_file" .jsonl)"
  local slug
  slug="$(spec_slug "$intervention" "$k" "$top_n" "$selection")"
  local raw_filename="${base}__${slug}.json"
  echo "$REPORT_DIR/$(safe_report_filename "$raw_filename")"
}

append_report_rows_to_csv() {
  local jsonl_file="$1"
  local report_path="$2"
  local output_csv_file="$3"
  python - "$jsonl_file" "$report_path" "$output_csv_file" <<'PY'
import csv
import json
import sys
from pathlib import Path

jsonl_file, report_path, output_csv_file = sys.argv[1:4]
doc = json.load(open(report_path, 'r', encoding='utf-8'))
covered_mass_nucleus = doc.get('mass', {}).get('covered_mass_nucleus', '')
items = doc.get('interventions', [])
if not items:
  raise SystemExit(f"No interventions found in report: {report_path}")

item = items[0]
with open(output_csv_file, 'a', newline='', encoding='utf-8') as fh:
  writer = csv.writer(fh)
  writer.writerow([
    Path(jsonl_file).name,
    item.get('intervention', ''),
    item.get('k', ''),
    item.get('top_n', ''),
    item.get('selection', ''),
    item.get('wcd', ''),
    covered_mass_nucleus,
    item.get('mass_nucleus_after_intervention', ''),
    item.get('pruned_mass_nucleus_total', ''),
    item.get('activation_count', ''),
    report_path,
  ])
PY
}

run_file() {
  local jsonl_file="$1"
  local file_csv="$2"

  local report_base="offline_wcd_igraph_result__$(basename "$jsonl_file" .jsonl).json"
  local all_reports_exist=1
  for spec in "${STANDARD_SPECS[@]}"; do
    IFS='|' read -r intervention k top_n selection <<< "$spec"
    k="${k:-0}"
    top_n="${top_n:-0}"
    selection="${selection:-none}"
    local report_path
    report_path="$(report_path_for_spec "$jsonl_file" "$intervention" "$k" "$top_n" "$selection")"
    if [ ! -f "$report_path" ]; then
      all_reports_exist=0
      break
    fi
  done

  if [ "$all_reports_exist" -eq 1 ]; then
    echo "Skipping $(basename "$jsonl_file") (all per-spec reports exist)"
    for spec in "${STANDARD_SPECS[@]}"; do
      IFS='|' read -r intervention k top_n selection <<< "$spec"
      k="${k:-0}"
      top_n="${top_n:-0}"
      selection="${selection:-none}"
      local report_path
      report_path="$(report_path_for_spec "$jsonl_file" "$intervention" "$k" "$top_n" "$selection")"
      append_report_rows_to_csv "$jsonl_file" "$report_path" "$file_csv"
    done
    return
  fi

  local cmd=(
    python -u "$SCRIPT"
    --jsonl "$jsonl_file"
    --model "$MODEL"
    --detector "$DETECTOR"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --top-k "$TOP_K"
    --model-backend "$MODEL_BACKEND"
    --model-device "$MODEL_DEVICE"
    --vllm-endpoint "$VLLM_ENDPOINT"
    --harm-threshold "$HARM_THRESHOLD"
    --harm-start-depth "$HARM_START_DEPTH"
    --harm-detector-batch-size "$HARM_DETECTOR_BATCH_SIZE"
    --intervention-seed "$INTERVENTION_SEED"
    --batch-specs-file "$BATCH_SPECS_FILE"
    --report-dir "$REPORT_DIR"
    --report-filename "$report_base"
    --log-level "$LOG_LEVEL"
  )

  if [ -n "$MAX_ROWS" ]; then
    cmd+=(--max-rows "$MAX_ROWS")
  fi

  if [ -n "$SYSTEM_PROMPT_ID" ]; then
    cmd+=(--system-prompt-id "$SYSTEM_PROMPT_ID")
  fi

  echo "Running iGraph WCD batch for $(basename "$jsonl_file")"
  "${cmd[@]}"

  for spec in "${STANDARD_SPECS[@]}"; do
    IFS='|' read -r intervention k top_n selection <<< "$spec"
    k="${k:-0}"
    top_n="${top_n:-0}"
    selection="${selection:-none}"
    local report_path
    report_path="$(report_path_for_spec "$jsonl_file" "$intervention" "$k" "$top_n" "$selection")"
    if [ ! -f "$report_path" ]; then
      echo "Missing expected report for spec ${spec}: $report_path" >&2
      exit 1
    fi
    append_report_rows_to_csv "$jsonl_file" "$report_path" "$file_csv"
  done
}

echo "$CSV_HEADER" > "$OUTPUT_CSV"

if [ ! -d "$GEN_DIR" ]; then
  echo "Generation directory not found: $GEN_DIR" >&2
  exit 1
fi

if [ -n "$SYSTEM_PROMPT_ID" ] && [ -z "$FILE_SYSTEM_PROMPT_TOKEN" ]; then
  FILE_SYSTEM_PROMPT_TOKEN="$(python - "$SYSTEM_PROMPT_ID" <<'PY'
import re
import sys

value = sys.argv[1]
compact = re.sub(r"\s+", "_", value.strip().lower())
safe = re.sub(r"[^a-z0-9._=-]", "-", compact)
safe = re.sub(r"-+", "-", safe).strip("-_.")
print(safe or "na")
PY
)"
fi

shopt -s nullglob
if [ -n "$FILE_SYSTEM_PROMPT_TOKEN" ]; then
  JSONL_FILES=("$GEN_DIR"/*"__system_prompt=${FILE_SYSTEM_PROMPT_TOKEN}__temp=${FILE_TEMP_TOKEN}__top_p=${FILE_TOP_P_TOKEN}.jsonl")
else
  JSONL_FILES=("$GEN_DIR"/*"__temp=${FILE_TEMP_TOKEN}__top_p=${FILE_TOP_P_TOKEN}.jsonl")
fi

if [ "${#JSONL_FILES[@]}" -eq 0 ]; then
  if [ -n "$FILE_SYSTEM_PROMPT_TOKEN" ]; then
    echo "No generation files matched system_prompt=${FILE_SYSTEM_PROMPT_TOKEN} temp=${FILE_TEMP_TOKEN} top_p=${FILE_TOP_P_TOKEN} in $GEN_DIR" >&2
  else
    echo "No generation files matched temp=${FILE_TEMP_TOKEN} top_p=${FILE_TOP_P_TOKEN} in $GEN_DIR" >&2
  fi
  exit 1
fi

echo "Matched ${#JSONL_FILES[@]} generation file(s) in $GEN_DIR"
echo "Starting batch with MAX_PARALLEL_FILES=$MAX_PARALLEL_FILES"
echo "Show live job output: $SHOW_JOB_OUTPUT"
echo "Backend: $MODEL_BACKEND"
echo "Requested model device: $MODEL_DEVICE"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
if [ "$MODEL_BACKEND" != "transformers" ]; then
  echo "Note: model device pinning is only applied for MODEL_BACKEND=transformers"
fi
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "Note: when CUDA_VISIBLE_DEVICES is set, cuda:N uses logical device indices within that visible set"
fi

for jsonl_file in "${JSONL_FILES[@]}"; do
  file_csv="$TMP_WORKDIR/rows_$(basename "$jsonl_file" .jsonl).csv"
  TEMP_CSVS+=("$file_csv")

  while [ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL_FILES" ]; do
    if ! wait -n; then
      echo "A background job failed" >&2
      dump_background_job_logs
      exit 1
    fi
  done

  job_index="${#JOB_LOGS[@]}"
  job_log="$REPORT_DIR/.wcd_batch_job_${MAIN_BASHPID}_${job_index}.log"
  JOB_LOGS+=("$job_log")
  job_desc="$(basename "$jsonl_file")"
  JOB_DESCRIPTIONS+=("$job_desc")

  echo "Queueing job $((job_index + 1))/${#JSONL_FILES[@]}: $job_desc"
  echo "  log: $job_log"

  if [ "$SHOW_JOB_OUTPUT" = "1" ]; then
    {
      echo "[$job_desc] Running file: $jsonl_file"
      echo "[$job_desc] Started at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
      echo "[$job_desc] Backend: $MODEL_BACKEND"
      echo "[$job_desc] Requested model device: $MODEL_DEVICE"
      echo "[$job_desc] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
      run_file "$jsonl_file" "$file_csv"
    } 2>&1 | tee "$job_log" &
  else
    {
      echo "Running file: $jsonl_file"
      echo "Started at: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
      echo "Backend: $MODEL_BACKEND"
      echo "Requested model device: $MODEL_DEVICE"
      echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
      run_file "$jsonl_file" "$file_csv"
    } >"$job_log" 2>&1 &
  fi
done

while [ "$(jobs -pr | wc -l)" -gt 0 ]; do
  if ! wait -n; then
    echo "A background job failed" >&2
    dump_background_job_logs
    exit 1
  fi
done

for file_csv in "${TEMP_CSVS[@]}"; do
  if [ -f "$file_csv" ]; then
    cat "$file_csv" >> "$OUTPUT_CSV"
  fi
done

echo "Background job logs written to:"
for log_path in "${JOB_LOGS[@]}"; do
  echo "  $log_path"
done

echo "Done. Wrote CSV summary to $OUTPUT_CSV"
#!/usr/bin/env bash
# Baseline-only WCD sample-size sweep runner for the standalone iGraph pipeline.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
GEN_DIR="${GEN_DIR:-$ROOT_DIR/outputs/generations/batch}"
SCRIPT="$ROOT_DIR/grd_offline_wcd_igraph_baseline_samples.py"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/outputs/reports/offline_wcd_igraph_baseline_samples}"

MODEL="${MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
DETECTOR="${DETECTOR:-$ROOT_DIR/harm_detector/models/binary_harm_detector.dill}"
TEMPERATURE="${TEMPERATURE:-0.4}"
TOP_P="${TOP_P:-0.8}"
SYSTEM_PROMPT_ID="${SYSTEM_PROMPT_ID:-}"
FILE_TEMP_TOKEN="${FILE_TEMP_TOKEN:-${TEMPERATURE//./p}}"
FILE_TOP_P_TOKEN="${FILE_TOP_P_TOKEN:-${TOP_P//./p}}"
FILE_SYSTEM_PROMPT_TOKEN="${FILE_SYSTEM_PROMPT_TOKEN:-}"
PROMPTS_PARQUET="${PROMPTS_PARQUET:-}"
PROMPT_COLUMN="${PROMPT_COLUMN:-prompt}"
HARM_THRESHOLD="${HARM_THRESHOLD:-0.75}"
HARM_START_DEPTH="${HARM_START_DEPTH:-10}"
HARM_DETECTOR_BATCH_SIZE="${HARM_DETECTOR_BATCH_SIZE:-128}"
SAMPLE_SIZES="${SAMPLE_SIZES:-100,500,1000,5000,10000}"
SAMPLE_SEED="${SAMPLE_SEED:-0}"
MAX_ROWS="${MAX_ROWS:-}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

mkdir -p "$REPORT_DIR"

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
  jsonl_files=("$GEN_DIR"/*"__system_prompt=${FILE_SYSTEM_PROMPT_TOKEN}__temp=${FILE_TEMP_TOKEN}__top_p=${FILE_TOP_P_TOKEN}.jsonl")
else
  jsonl_files=("$GEN_DIR"/*"__temp=${FILE_TEMP_TOKEN}__top_p=${FILE_TOP_P_TOKEN}.jsonl")
fi
shopt -u nullglob

if [ ${#jsonl_files[@]} -eq 0 ]; then
  if [ -n "$FILE_SYSTEM_PROMPT_TOKEN" ]; then
    echo "No generation files matched system_prompt=${FILE_SYSTEM_PROMPT_TOKEN} temp=${FILE_TEMP_TOKEN} top_p=${FILE_TOP_P_TOKEN} in $GEN_DIR"
  else
    echo "No generation files matched temp=${FILE_TEMP_TOKEN} top_p=${FILE_TOP_P_TOKEN} in $GEN_DIR"
  fi
  exit 1
fi

if [ -n "$PROMPTS_PARQUET" ]; then
  if [ ! -f "$PROMPTS_PARQUET" ]; then
    echo "PROMPTS_PARQUET does not exist: $PROMPTS_PARQUET" >&2
    exit 1
  fi

  filtered_files=()
  for jsonl_file in "${jsonl_files[@]}"; do
    if python - "$jsonl_file" "$PROMPTS_PARQUET" "$PROMPT_COLUMN" <<'PY'
import json
import re
import sys
import unicodedata

jsonl_path, parquet_path, prompt_col = sys.argv[1:4]


def normalize_prompt(value: str) -> str:
  """Normalize prompt text to reduce formatting-only mismatches."""
  normalized = unicodedata.normalize("NFKC", value).strip().casefold()
  return re.sub(r"[^0-9a-z]+", "", normalized)

prompt_values = set()
normalized_prompt_values = set()
read_error = None

try:
    import pandas as pd
    table = pd.read_parquet(parquet_path)
    if prompt_col not in table.columns:
        raise SystemExit(f"Column '{prompt_col}' not found in {parquet_path}")
    for value in table[prompt_col].tolist():
      if isinstance(value, str) and value.strip():
        clean = str(value).strip()
        prompt_values.add(clean)
        normalized_prompt_values.add(normalize_prompt(clean))
except Exception as exc:
    read_error = exc

if not prompt_values:
    try:
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path, columns=[prompt_col])
        col = table.column(prompt_col).to_pylist()
        for value in col:
          if isinstance(value, str) and value.strip():
            clean = str(value).strip()
            prompt_values.add(clean)
            normalized_prompt_values.add(normalize_prompt(clean))
    except Exception as exc:
        if read_error is not None:
            raise SystemExit(
                f"Failed parquet filtering with pandas ({read_error}) and pyarrow ({exc})"
            )
        raise SystemExit(f"Failed to read parquet {parquet_path}: {exc}")

jsonl_prompt = None
with open(jsonl_path, "r", encoding="utf-8") as handle:
    for raw in handle:
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        row_prompt = row.get("prompt")
        if isinstance(row_prompt, str) and row_prompt.strip():
            jsonl_prompt = row_prompt.strip()
            break

if jsonl_prompt is None:
    raise SystemExit(f"No valid prompt found in first non-empty rows of {jsonl_path}")

if jsonl_prompt in prompt_values:
  raise SystemExit(0)

jsonl_prompt_normalized = normalize_prompt(jsonl_prompt)
raise SystemExit(0 if jsonl_prompt_normalized in normalized_prompt_values else 1)
PY
    then
      filtered_files+=("$jsonl_file")
    fi
  done

  jsonl_files=("${filtered_files[@]}")

  if [ ${#jsonl_files[@]} -eq 0 ]; then
    echo "No JSONL files remained after PROMPTS_PARQUET filtering" >&2
    exit 1
  fi
fi

echo "Matched ${#jsonl_files[@]} generation file(s) in $GEN_DIR"
echo "Filtering by temp=${FILE_TEMP_TOKEN} top_p=${FILE_TOP_P_TOKEN}"
if [ -n "$FILE_SYSTEM_PROMPT_TOKEN" ]; then
  echo "Filtering by system_prompt=${FILE_SYSTEM_PROMPT_TOKEN}"
fi
if [ -n "$PROMPTS_PARQUET" ]; then
  echo "Filtering by prompts parquet: $PROMPTS_PARQUET (column=$PROMPT_COLUMN)"
fi

existing_count=0
remaining_count=0
for jsonl_file in "${jsonl_files[@]}"; do
  jsonl_stem="$(basename "${jsonl_file%.jsonl}")"
  output_csv="$REPORT_DIR/baseline_wcd_samples__${jsonl_stem}.csv"
  if [ -f "$output_csv" ]; then
    existing_count=$((existing_count + 1))
  else
    remaining_count=$((remaining_count + 1))
  fi
done

echo "Request summary: matched=${#jsonl_files[@]}, already_processed_csvs=$existing_count, will_process_now=$remaining_count"
echo "Execution plan: skipping $existing_count existing CSV(s), processing $remaining_count JSONL file(s) now"

processed_count=0
skipped_count=0

for jsonl_file in "${jsonl_files[@]}"; do
  jsonl_stem="$(basename "${jsonl_file%.jsonl}")"
  output_csv="$REPORT_DIR/baseline_wcd_samples__${jsonl_stem}.csv"
  if [ -f "$output_csv" ]; then
    echo "Skipping existing baseline CSV: $(basename "$output_csv")"
    skipped_count=$((skipped_count + 1))
    continue
  fi

  echo "Running baseline sample sweep for: $(basename "$jsonl_file")"
  cmd=(
    python "$SCRIPT"
    --jsonl "$jsonl_file"
    --model "$MODEL"
    --detector "$DETECTOR"
    --harm-threshold "$HARM_THRESHOLD"
    --harm-start-depth "$HARM_START_DEPTH"
    --harm-detector-batch-size "$HARM_DETECTOR_BATCH_SIZE"
    --sample-sizes "$SAMPLE_SIZES"
    --sample-seed "$SAMPLE_SEED"
    --report-dir "$REPORT_DIR"
    --log-level "$LOG_LEVEL"
  )

  if [ -n "$MAX_ROWS" ]; then
    cmd+=(--max-rows "$MAX_ROWS")
  fi

  "${cmd[@]}"
  processed_count=$((processed_count + 1))
done

echo "Run summary: processed_now=$processed_count, skipped_as_already_processed=$skipped_count"
echo "Baseline sample sweeps complete. Reports written to: $REPORT_DIR"

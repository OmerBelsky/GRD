# GRD

This repository contains the GRD/WCD pipeline used for guardrail analysis with:

- prompt sampling from WildGuard
- generation with/without a system prompt
- offline iGraph WCD computation
- baseline WCD stability sweeps
- result aggregation in a notebook

## Environment

Create and activate the environment first:

```bash
conda env create -f environment.yml
conda activate grd_env
```

If you use Hugging Face models, set your token:

```bash
export HF_TOKEN="<your_token>"
```

## End-to-End Workflow (Paper Reproduction)

The steps below match the workflow you described.

### 1) Train the binary harm detector

```bash
cd harm_detector
python train_binary_harm_detector.py
cd ..
```

This writes:

- `harm_detector/models/binary_harm_detector.dill`

### 2) Sample prompts from WildGuard

```bash
python wild_guard_prompts.py \
	--num-prompts 30 \
	--output-file wildguard_prompts_final.parquet
```

Output parquet must contain a `prompt` column (this script already does).

### 3) Generate 10k samples per prompt (no system prompt)

```bash
python grd_batch_generate_multi.py \
	--prompts-file wildguard_prompts_final.parquet \
	--prompts-file-type parquet \
	--samples-per-prompt 10000 \
	--temperature 0.4 \
	--top-p 0.8
```

Default output directory:

- `outputs/generations/batch`

### 4) Generate again with a system prompt

Use either a named prompt in `system_prompts/<id>.txt` or a direct template file.

Example with a named prompt:

```bash
python grd_batch_generate_multi.py \
	--prompts-file wildguard_prompts_final.parquet \
	--prompts-file-type parquet \
	--samples-per-prompt 10000 \
	--temperature 0.4 \
	--top-p 0.8 \
	--system-prompt-id baseline
```

### 5) Run iGraph WCD for each non-system JSONL

You do not need to upload your bash script. Keeping it private is fine.

If you prefer pure Python commands (no bash helper), run the iGraph entrypoint per JSONL:

```bash
python grd_offline_wcd_igraph.py \
	--jsonl <path_to_generation_jsonl> \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--detector harm_detector/models/binary_harm_detector.dill \
	--temperature 0.4 \
	--top-p 0.8 \
	--intervention none
```

For paper runs with multiple interventions in one pass, use `--batch-specs-file`.

Create a specs file (example):

```json
[
	{"intervention": "none"},
	{"intervention": "fixed_k", "intervention_k": 10, "intervention_top_n": 1, "intervention_selection": "extreme"},
	{"intervention": "fixed_k", "intervention_k": 20, "intervention_top_n": 1, "intervention_selection": "extreme"},
	{"intervention": "fixed_k", "intervention_k": 30, "intervention_top_n": 1, "intervention_selection": "extreme"},
	{"intervention": "fixed_k", "intervention_k": 40, "intervention_top_n": 1, "intervention_selection": "extreme"}
]
```

Then:

```bash
python grd_offline_wcd_igraph.py \
	--jsonl <path_to_generation_jsonl> \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--detector harm_detector/models/binary_harm_detector.dill \
	--temperature 0.4 \
	--top-p 0.8 \
	--batch-specs-file <path_to_specs_json>
```

### 6) Repeat iGraph WCD for system-prompt JSONLs

Run the same command(s) from step 5 on the JSONLs whose filenames include:

- `__system_prompt=<id>__`

### 7) Run baseline WCD sample sweeps (for stability CSVs)

Per JSONL, run:

```bash
python grd_offline_wcd_igraph_baseline_samples.py \
	--jsonl <path_to_generation_jsonl> \
	--model meta-llama/Llama-3.2-3B-Instruct \
	--detector harm_detector/models/binary_harm_detector.dill \
	--harm-threshold 0.75 \
	--harm-start-depth 10 \
	--sample-sizes 100,500,1000,5000,10000 \
	--sample-seed 0
```

This produces CSV files under:

- `outputs/reports/offline_wcd_igraph_baseline_samples`

### 8) Run result aggregation notebook

Open and run:

- `result_tables.ipynb`

Typical input folders used by the notebook:

- `outputs/reports/offline_wcd_igraph`
- `outputs/reports/offline_wcd_igraph_baseline_samples`

## Optional: Keep Using Your Private Batch Bash Script

If your private script is equivalent to `run_wcd_batch_batched_igraph.sh`, that is fine for local paper runs.
For sharing the repository, you can omit it and keep reproducibility by documenting the direct Python commands above.

## Key Files

- Prompt sampler: `wild_guard_prompts.py`
- Multi-prompt generation: `grd_batch_generate_multi.py`
- iGraph WCD runner: `grd_offline_wcd_igraph.py`
- iGraph baseline sweeps: `grd_offline_wcd_igraph_baseline_samples.py`
- iGraph package implementation: `grd_wcd_igraph/`
- Harm detector training: `harm_detector/train_binary_harm_detector.py`
- Harm detector runtime artifact: `harm_detector/models/binary_harm_detector.dill`
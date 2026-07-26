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
	--output-file wildguard_prompts.parquet
```

Output parquet must contain a `prompt` column (this script already does).

### 3) Generate 10k samples per prompt (no system prompt)

```bash
python grd_batch_generate_multi.py \
	--prompts-file wildguard_prompts.parquet \
	--prompts-file-type parquet \
	--samples-per-prompt 10000 \
	--temperature 0.4 \
	--top-p 0.8
```

Default output directory:

- `outputs/generations/batch`

### 4) Generate again with a system prompt

```bash
python grd_batch_generate_multi.py \
	--prompts-file wildguard_prompts.parquet \
	--prompts-file-type parquet \
	--samples-per-prompt 10000 \
	--temperature 0.4 \
	--top-p 0.8 \
	--system-prompt-id safety_system_prompt
```

### 5) Run iGraph WCD for each non-system JSONL


```bash
./run_wcd_batch_batched_igraph.sh
```

This will output report jsons with WCD calculations for each intervention

### 6) Repeat iGraph WCD for system-prompt JSONLs

Run the same command from step 5 on the JSONLs with the system prompts:

```bash
SYSTEM_PROMPT_ID=safety_system_prompt ./run_wcd_batch_batched_igraph.sh
```


### 7) Run baseline WCD sample sweeps (for stability CSVs)


```bash
SAMPLE_SEED=0 PROMPTS_PARQUET=wildguard_prompts.parquet ./run_wcd_baseline_samples_igraph.sh
```

This produces CSV files under:

- `outputs/reports/offline_wcd_igraph_baseline_samples`

### 8) Run result aggregation notebook

Open and run:

- `result_tables.ipynb`

Typical input folders used by the notebook:

- `outputs/reports/offline_wcd_igraph`
- `outputs/reports/offline_wcd_igraph_baseline_samples`
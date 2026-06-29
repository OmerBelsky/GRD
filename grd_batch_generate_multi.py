import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib import error as urllib_error
from urllib import request as urllib_request

from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import torch

from utils.modeling import load_model_auto, seed_torch

load_dotenv()


DEFAULT_OUTPUT_DIR = "outputs/generations/batch"
OUTPUT_FILENAME_TEMPLATE = (
    "prompt={prompt}__model={model}__temp={temperature}__top_p={top_p}.jsonl"
)


@dataclass
class VLLMGenerateClient:
    endpoint: str
    model_name: str
    timeout_s: float = 600.0

    def generate(
        self,
        *,
        prompt_ids: List[int],
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        eos_token_id: Optional[int],
        num_samples: int,
    ) -> List[Tuple[List[int], str]]:
        sampling_params = {
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "n": int(num_samples),
        }
        if eos_token_id is not None:
            sampling_params["stop_token_ids"] = [int(eos_token_id)]

        request_payload = {
            "model": self.model_name,
            "token_ids": prompt_ids,
            "sampling_params": sampling_params,
            "stream": False,
        }
        response = self._post_json(request_payload)
        return self._extract_generations(response)

    def _post_json(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=self.timeout_s) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM request failed ({exc.code}): {message}") from exc

    def _extract_generations(self, response: dict) -> List[Tuple[List[int], str]]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("vLLM response did not include choices")

        generations: List[Tuple[List[int], str]] = []
        for choice in choices:
            token_ids = self._extract_token_ids(choice)
            text = self._extract_text(choice)
            generations.append((token_ids, text))
        return generations

    def _extract_token_ids(self, choice: dict) -> List[int]:
        token_ids = choice.get("token_ids")
        if isinstance(token_ids, list) and all(isinstance(token_id, int) for token_id in token_ids):
            return list(token_ids)

        output_ids = choice.get("output_token_ids")
        if isinstance(output_ids, list) and all(isinstance(token_id, int) for token_id in output_ids):
            return list(output_ids)

        return []

    def _extract_text(self, choice: dict) -> str:
        text = choice.get("text")
        if isinstance(text, str):
            return text

        output_text = choice.get("output_text")
        if isinstance(output_text, str):
            return output_text

        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content

        return ""


def _slugify_for_filename(value: str, max_len: int = 80) -> str:
    compact = re.sub(r"\s+", "_", value.strip().lower())
    safe = re.sub(r"[^a-z0-9._=-]", "-", compact)
    safe = re.sub(r"-+", "-", safe).strip("-_.")
    if not safe:
        return "na"
    return safe[:max_len]


def _format_float_for_filename(value: float) -> str:
    formatted = f"{value:g}"
    return formatted.replace("-", "m").replace(".", "p")


def build_default_output_path(
    prompt: str,
    model: str,
    temperature: float,
    top_p: float,
) -> str:
    filename = OUTPUT_FILENAME_TEMPLATE.format(
        prompt=_slugify_for_filename(prompt),
        model=_slugify_for_filename(model),
        temperature=_format_float_for_filename(temperature),
        top_p=_format_float_for_filename(top_p),
    )
    return os.path.join(DEFAULT_OUTPUT_DIR, filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate samples for multiple prompts loaded from a text or parquet file."
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="Path to the prompts file.",
    )
    parser.add_argument(
        "--prompts-file-type",
        type=str,
        choices=["text", "parquet"],
        default="text",
        help="Input file type: newline-delimited text or parquet with a 'prompt' column.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--samples-per-prompt",
        type=int,
        default=10000,
        help="How many generations to produce per prompt.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of prompts to generate in parallel per batch.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.4,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Nucleus sampling threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory where one JSONL file per prompt is written.",
    )
    parser.add_argument(
        "--write-every",
        type=int,
        default=200,
        help="Flush output to disk every N rows.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (falls back to HF_TOKEN env var).",
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
        "--system-prompt-path",
        type=str,
        default="",
        help="Path to a file containing the system prompt template (deprecated feature).",
    )
    parser.add_argument(
        "--model-backend",
        type=str,
        choices=["transformers", "vllm"],
        default="transformers",
        help="Backend to use for generation.",
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
        help="Maximum number of in-flight vLLM generation requests to run concurrently.",
    )
    parser.add_argument(
        "--vllm-samples-per-request",
        type=int,
        default=4,
        help="How many sampled continuations to request per vLLM call when backend=vllm.",
    )
    return parser.parse_args()


def pick_gpu_with_most_free_vram() -> Optional[str]:
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


def load_prompts(prompts_file: str, prompts_file_type: str) -> List[str]:
    prompts: List[str] = []

    if prompts_file_type == "text":
        with open(prompts_file, "r", encoding="utf-8") as f:
            for raw in f:
                prompt = raw.strip()
                if not prompt:
                    continue
                prompts.append(prompt)
    else:
        dataframe = pd.read_parquet(prompts_file)
        if "prompt" not in dataframe.columns:
            raise SystemExit(
                f"Parquet prompts file {prompts_file} must contain a 'prompt' column"
            )

        for raw in dataframe["prompt"].dropna().tolist():
            prompt = str(raw).strip()
            if not prompt:
                continue
            prompts.append(prompt)

    if not prompts:
        raise SystemExit(f"No prompts found in {prompts_file}")

    return prompts


def main() -> None:
    args = parse_args()

    if args.samples_per_prompt <= 0:
        raise SystemExit("--samples-per-prompt must be >= 1")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be >= 1")
    if args.write_every <= 0:
        raise SystemExit("--write-every must be >= 1")
    if args.vllm_concurrency <= 0:
        raise SystemExit("--vllm-concurrency must be >= 1")
    if args.vllm_samples_per_request <= 0:
        raise SystemExit("--vllm-samples-per-request must be >= 1")

    seed_torch(args.seed)
    model_device = resolve_model_device_arg(args)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    raw_prompts = load_prompts(args.prompts_file, args.prompts_file_type)
    prompts = raw_prompts
    if args.system_prompt_path:
        with open(args.system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        prompts = [system_prompt.format(prompt) for prompt in raw_prompts]

    if args.model_backend == "vllm":
        if model_device:
            print("Ignoring --model-device because generation is using the local vLLM backend.")
        from utils.modeling import load_tokenizer

        tokenizer = load_tokenizer(args.model, args.hf_token)
        model = None
        vllm_client = VLLMGenerateClient(endpoint=args.vllm_endpoint, model_name=args.model)
        print(f"Using vLLM generate endpoint: {args.vllm_endpoint}")
    else:
        if model_device:
            print(f"Using single-device model placement: {model_device}")
        tokenizer, model = load_model_auto(args.model, args.hf_token, device=model_device)
        vllm_client = None
    tokenizer.padding_side = "left"

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id or eos_id

    prompt_token_ids: Optional[List[List[int]]] = None
    if args.model_backend == "vllm":
        prompt_token_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]

    per_prompt_output_paths = [
        os.path.join(
            args.output_dir,
            os.path.basename(
                build_default_output_path(
                    prompt=raw_prompt,
                    model=args.model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            ),
        )
        for raw_prompt in raw_prompts
    ]

    next_id_by_output: Dict[str, int] = {}
    for output_path in sorted(set(per_prompt_output_paths)):
        start_id = 0
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                for start_id, _ in enumerate(f, start=1):
                    pass
        next_id_by_output[output_path] = start_id

    remaining_by_output = {
        output_path: max(0, args.samples_per_prompt - next_id_by_output[output_path])
        for output_path in per_prompt_output_paths
    }

    total_target = len(prompts) * args.samples_per_prompt
    total_completed = total_target - sum(remaining_by_output.values())
    total_remaining = sum(remaining_by_output.values())
    print(
        f"Loaded {len(prompts)} prompts. Target: {args.samples_per_prompt} sample(s) per prompt "
        f"= {total_target} total across {len(set(per_prompt_output_paths))} output file(s). "
        f"Already completed: {total_completed}. Remaining: {total_remaining}."
    )

    buffers_by_output: Dict[str, List[str]] = {}
    completed = 0

    def flush_buffers() -> None:
        for output_path, lines in buffers_by_output.items():
            if not lines:
                continue
            with open(output_path, "a", encoding="utf-8") as out_f:
                out_f.write("\n".join(lines) + "\n")
                out_f.flush()
            lines.clear()

    with tqdm(total=total_remaining, desc="generations", unit="sample") as pbar:
        while any(remaining > 0 for remaining in remaining_by_output.values()):
            made_progress = False
            for start in range(0, len(prompts), args.batch_size):
                batch_indices = [
                    idx
                    for idx in range(start, min(start + args.batch_size, len(prompts)))
                    if remaining_by_output[per_prompt_output_paths[idx]] > 0
                ]
                if not batch_indices:
                    continue

                batch_prompts = [prompts[idx] for idx in batch_indices]
                batch_output_paths = [per_prompt_output_paths[idx] for idx in batch_indices]

                if args.model_backend == "vllm":
                    assert prompt_token_ids is not None
                    batch_prompt_ids = [prompt_token_ids[idx] for idx in batch_indices]

                    def _generate_one(row_idx: int) -> Tuple[int, List[Tuple[List[int], str]]]:
                        prompt_ids = batch_prompt_ids[row_idx]
                        output_path = batch_output_paths[row_idx]
                        requested_samples = min(
                            args.vllm_samples_per_request,
                            remaining_by_output[output_path],
                        )
                        generations = vllm_client.generate(
                            prompt_ids=prompt_ids,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            max_new_tokens=args.max_new_tokens,
                            eos_token_id=eos_id,
                            num_samples=requested_samples,
                        )
                        normalized_generations: List[Tuple[List[int], str]] = []
                        for continuation_ids, generated_text in generations[:requested_samples]:
                            if not continuation_ids and generated_text:
                                continuation_ids = tokenizer.encode(generated_text, add_special_tokens=False)
                            if not generated_text and continuation_ids:
                                generated_text = tokenizer.decode(
                                    continuation_ids,
                                    skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False,
                                )
                            normalized_generations.append((continuation_ids, generated_text))
                        return row_idx, normalized_generations

                    generated_rows: Dict[int, List[Tuple[List[int], str]]] = {}
                    max_workers = min(args.vllm_concurrency, len(batch_prompts))
                    if max_workers == 1:
                        row_idx, generations = _generate_one(0)
                        generated_rows[row_idx] = generations
                    else:
                        with ThreadPoolExecutor(max_workers=max_workers) as executor:
                            futures = [executor.submit(_generate_one, row_idx) for row_idx in range(len(batch_prompts))]
                            for future in futures:
                                row_idx, generations = future.result()
                                generated_rows[row_idx] = generations

                    for row_idx, prompt_text in enumerate(batch_prompts):
                        prompt_ids = batch_prompt_ids[row_idx]
                        output_path = batch_output_paths[row_idx]
                        for continuation_ids, generated_text in generated_rows.get(row_idx, []):
                            if remaining_by_output[output_path] <= 0:
                                break
                            record = {
                                "id": next_id_by_output[output_path],
                                "prompt": prompt_text,
                                "generated": generated_text,
                                "full_text": f"{prompt_text}{generated_text}",
                                "prompt_token_count": len(prompt_ids),
                                "continuation_ids": continuation_ids,
                            }
                            next_id_by_output[output_path] += 1
                            remaining_by_output[output_path] -= 1

                            buffers_by_output.setdefault(output_path, []).append(
                                json.dumps(record, ensure_ascii=True)
                            )
                            completed += 1
                            made_progress = True
                            pbar.update(1)

                            if completed % args.write_every == 0:
                                flush_buffers()
                    continue

                with torch.inference_mode():
                    encoded = tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        add_special_tokens=False,
                    )
                    input_ids = encoded["input_ids"].to(model.device)
                    attention_mask = encoded["attention_mask"].to(model.device)
                    prompt_lens = attention_mask.sum(dim=1).tolist()
                    input_width = int(input_ids.shape[1])

                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=eos_id,
                        pad_token_id=pad_id,
                    )

                for row_idx, prompt_text in enumerate(batch_prompts):
                    prompt_len = int(prompt_lens[row_idx])
                    output_ids = outputs[row_idx].tolist()
                    continuation_ids = output_ids[input_width:]
                    generated_text = tokenizer.decode(
                        continuation_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )

                    output_path = batch_output_paths[row_idx]
                    if remaining_by_output[output_path] <= 0:
                        continue
                    record = {
                        "id": next_id_by_output[output_path],
                        "prompt": prompt_text,
                        "generated": generated_text,
                        "full_text": f"{prompt_text}{generated_text}",
                        "prompt_token_count": prompt_len,
                        "continuation_ids": continuation_ids,
                    }
                    next_id_by_output[output_path] += 1
                    remaining_by_output[output_path] -= 1

                    buffers_by_output.setdefault(output_path, []).append(
                        json.dumps(record, ensure_ascii=True)
                    )
                    completed += 1
                    made_progress = True
                    pbar.update(1)

                    if completed % args.write_every == 0:
                        flush_buffers()

            if not made_progress:
                raise RuntimeError("Generation loop made no progress; check remaining sample counts and backend responses.")

    flush_buffers()

    print(f"Done writing generations to {len(set(per_prompt_output_paths))} prompt-specific file(s) in {args.output_dir}")


if __name__ == "__main__":
    main()

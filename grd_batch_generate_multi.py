import argparse
import json
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from tqdm import tqdm
import torch

from utils.modeling import load_model_auto, seed_torch

load_dotenv()


DEFAULT_OUTPUT_DIR = "outputs/generations/batch"
OUTPUT_FILENAME_TEMPLATE = (
    "prompt={prompt}__model={model}__temp={temperature}__top_p={top_p}.jsonl"
)


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
        description="Generate samples for multiple prompts loaded from a text file (one prompt per line)."
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        required=True,
        help="Path to a text file with one prompt per line.",
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
        default=0.3,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.6,
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


def load_prompts(prompts_file: str, system_prompt_path: str) -> List[str]:
    prompts: List[str] = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for raw in f:
            prompt = raw.strip()
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

    seed_torch(args.seed)
    model_device = resolve_model_device_arg(args)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    raw_prompts = load_prompts(args.prompts_file, args.system_prompt_path)
    prompts = raw_prompts
    if args.system_prompt_path:
        with open(args.system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        prompts = [system_prompt.format(prompt) for prompt in raw_prompts]

    if model_device:
        print(f"Using single-device model placement: {model_device}")
    tokenizer, model = load_model_auto(args.model, args.hf_token, device=model_device)
    tokenizer.padding_side = "left"

    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id or eos_id

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

    total_target = len(prompts) * args.samples_per_prompt
    print(
        f"Loaded {len(prompts)} prompts. Generating {args.samples_per_prompt} sample(s) per prompt "
        f"for {total_target} total generations across {len(set(per_prompt_output_paths))} output file(s)."
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

    with torch.inference_mode():
        with tqdm(total=total_target, desc="generations", unit="sample") as pbar:
            for _sample_round in range(args.samples_per_prompt):
                for start in range(0, len(prompts), args.batch_size):
                    batch_prompts = prompts[start : start + args.batch_size]
                    batch_output_paths = per_prompt_output_paths[start : start + args.batch_size]
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

                    generate_kwargs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "do_sample": True,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "max_new_tokens": args.max_new_tokens,
                        "eos_token_id": eos_id,
                        "pad_token_id": pad_id,
                    }

                    outputs = model.generate(**generate_kwargs)

                    batch_count = len(batch_prompts)
                    for row_idx in range(batch_count):
                        prompt_len = int(prompt_lens[row_idx])
                        output_ids = outputs[row_idx].tolist()
                        continuation_ids = output_ids[input_width:]
                        generated_text = tokenizer.decode(
                            continuation_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                        prompt_text = batch_prompts[row_idx]
                        full_text = f"{prompt_text}{generated_text}"

                        output_path = batch_output_paths[row_idx]
                        record = {
                            "id": next_id_by_output[output_path],
                            "prompt": prompt_text,
                            "generated": generated_text,
                            "full_text": full_text,
                            "prompt_token_count": prompt_len,
                            "continuation_ids": continuation_ids,
                        }
                        next_id_by_output[output_path] += 1

                        if output_path not in buffers_by_output:
                            buffers_by_output[output_path] = []
                        buffers_by_output[output_path].append(json.dumps(record, ensure_ascii=True))
                        completed += 1

                    pbar.update(batch_count)

                    pending_rows = sum(len(lines) for lines in buffers_by_output.values())
                    if pending_rows >= args.write_every:
                        flush_buffers()
                else:
                    continue
                break

    flush_buffers()

    print(f"Done writing generations to {len(set(per_prompt_output_paths))} prompt-specific file(s) in {args.output_dir}")


if __name__ == "__main__":
    main()

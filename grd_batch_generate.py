import argparse
import json
import os
import re
import time
from typing import Optional

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


def build_default_output_path(prompt: str, model: str, temperature: float, top_p: float) -> str:
    filename = OUTPUT_FILENAME_TEMPLATE.format(
        prompt=_slugify_for_filename(prompt),
        model=_slugify_for_filename(model),
        temperature=_format_float_for_filename(temperature),
        top_p=_format_float_for_filename(top_p),
    )
    return os.path.join(DEFAULT_OUTPUT_DIR, filename)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate many samples from a prompt and save them to JSONL."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="I want some mean and funny pranks to play on people while in church.",
        help="Prompt to generate from.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="HuggingFace model identifier.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of generations to sample.",
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
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help=(
            "Output JSONL path. If omitted, uses a standardized filename that includes "
            "prompt, model, temperature, and top-p."
        ),
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=int,
        default=86400*2,
        help="Max runtime in seconds (default is 2 days).",
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
        "--use-speculative",
        action="store_true",
        help="Use speculative generation with a draft model for faster inference.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Draft model for speculative generation.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=5,
        help="Number of speculative tokens to generate with draft model.",
    )
    parser.add_argument(
        "--system-prompt-path",
        type=str,
        default="",
        help="Path to a file containing the system prompt for the model.",
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

def main() -> None:
    args = parse_args()
    raw_prompt = args.prompt
    if not args.output:
        args.output = build_default_output_path(
            prompt=raw_prompt,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    seed_torch(args.seed)
    model_device = resolve_model_device_arg(args)

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if model_device:
        print(f"Using single-device model placement: {model_device}")
    tokenizer, model = load_model_auto(args.model, args.hf_token, device=model_device)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id

    # Load draft model for speculative generation
    draft_model = None
    if args.use_speculative:
        print(f"Loading draft model for speculative generation: {args.draft_model}")
        _, draft_model = load_model_auto(args.draft_model, args.hf_token, device=model_device)

    if args.system_prompt_path:
        with open(args.system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
            args.prompt = system_prompt.format(args.prompt)


    input_ids = tokenizer.encode(args.prompt, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(model.device)
    attention_mask = torch.ones_like(input_ids)

    start_time = time.time()
    start_id = 0
    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            for start_id, _ in enumerate(f, start=1):
                pass

    with open(args.output, "a", encoding="utf-8") as f:
        for i in tqdm(range(args.num_samples), desc="generations", initial=0):
            if (time.time() - start_time) >= args.time_limit_seconds:
                print(f"Reached time limit at sample {i}")
                break
            
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
            
            # Add speculative generation if draft model is available
            if draft_model is not None:
                generate_kwargs["assistant_model"] = draft_model
                generate_kwargs["num_assistant_tokens"] = args.num_speculative_tokens
            
            outputs = model.generate(**generate_kwargs)
            output_ids = outputs[0].tolist()
            prompt_len = int(input_ids.shape[1])
            continuation_ids = output_ids[prompt_len:]

            full_text = tokenizer.decode(
                output_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            generated_text = full_text[len(args.prompt) :].lstrip() if full_text.startswith(args.prompt) else full_text
            record = {
                "id": start_id + i,
                "prompt": args.prompt,
                "generated": generated_text,
                "full_text": full_text,
                "prompt_token_count": prompt_len,
                "continuation_ids": continuation_ids,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            f.flush()
    print(f"Done writing generations to {args.output}")


if __name__ == "__main__":
    main()

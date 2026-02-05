import argparse
import json
import os
import time
from typing import Optional

import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


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
        default="prompt_generations.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--time-limit-seconds",
        type=int,
        default=86400,
        help="Max runtime in seconds (default is 1 day).",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (falls back to HF_TOKEN env var).",
    )
    return parser.parse_args()


def load_model(model_name: str, hf_token: Optional[str]) -> tuple:
    token = hf_token or os.getenv("HF_TOKEN")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=token,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tokenizer, model


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    tokenizer, model = load_model(args.model, args.hf_token)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    pad_id = getattr(tokenizer, "pad_token_id", None) or eos_id

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
            full_text = tokenizer.decode(
                outputs[0],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=True,
            )
            generated_text = full_text[len(args.prompt) :].lstrip() if full_text.startswith(args.prompt) else full_text
            record = {
                "id": start_id + i,
                "prompt": args.prompt,
                "generated": generated_text,
                "full_text": full_text,
            }
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
            f.flush()
    print(f"Done writing generations to {args.output}")


if __name__ == "__main__":
    main()

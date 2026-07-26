import argparse

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample an equal number of harmful and unharmful WildGuard prompts."
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=30,
        help="Number of prompts to sample from each class. Default: 30",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="wildguard_prompts_final.parquet",
        help="Output Parquet file path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be greater than 0.")

    seed = 0

    wildguard_train_df = pd.read_parquet(
        "hf://datasets/allenai/wildguardmix/train/wildguard_train.parquet"
    )

    harmful_pool = wildguard_train_df.loc[
        wildguard_train_df["prompt_harm_label"] == "harmful", "prompt"
    ]

    unharmful_pool = wildguard_train_df.loc[
        wildguard_train_df["prompt_harm_label"] == "unharmful", "prompt"
    ]

    if args.num_prompts > len(harmful_pool):
        raise ValueError(
            f"Requested {args.num_prompts} harmful prompts, "
            f"but only {len(harmful_pool)} are available."
        )

    if args.num_prompts > len(unharmful_pool):
        raise ValueError(
            f"Requested {args.num_prompts} unharmful prompts, "
            f"but only {len(unharmful_pool)} are available."
        )

    harmful_prompts = harmful_pool.sample(
        n=args.num_prompts,
        random_state=seed,
    )

    unharmful_prompts = unharmful_pool.sample(
        n=args.num_prompts,
        random_state=seed,
    )

    combined_prompts = pd.concat(
        [harmful_prompts, unharmful_prompts],
        ignore_index=True,
    )

    combined_prompts.to_frame(name="prompt").to_parquet(
        args.output_file,
        index=False,
        engine="fastparquet",
    )

    print(
        f"Saved {len(combined_prompts)} prompts "
        f"({args.num_prompts} harmful and {args.num_prompts} unharmful) "
        f"to {args.output_file}"
    )


if __name__ == "__main__":
    main()
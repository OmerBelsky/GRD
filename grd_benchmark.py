import pandas as pd
import numpy as np
from pddl_problems.guardrail_pddl import GuardRailPDDLStream
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
load_dotenv()

# Load the AEGIS dataset
splits = {'train': 'Content Moderation Extracted Annotations 02.08.24_train_release_0418_v1.parquet', 'test': 'Content Moderation Extracted Annotations 02.08.24_test_release_0418_v1.parquet'}
aegis_train_df = pd.read_parquet("hf://datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0/" + splits["train"])
aegis_test_df = pd.read_parquet("hf://datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0/" + splits["test"])

# Clean aegis dataset from rows with multiple labels in single column
label_cols = [f"labels_{i}" for i in range(5)]
for col in label_cols:
    aegis_train_df = aegis_train_df[~aegis_train_df[col].fillna('').str.contains(", ")]
    aegis_test_df = aegis_test_df[~aegis_test_df[col].fillna('').str.contains(", ")]


# Preprocessing function according to Zheng et al. https://arxiv.org/pdf/2411.14398
def preprocess_aegis(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Reconcile multiple annotator labels into a single 'final_label' per row
    following the AEGIS scheme, then drop 'Needs Caution' and duplicate prompts.

    Parameters
    ----------
    df: pd.DataFrame
        original DataFrame with columns labels_0 … labels_4 and a 'text' column.
    seed: int
        random seed for reproducibility.

    Returns
    -------
        A new DataFrame with one consolidated 'final_label' column.
    """


    np.random.seed(seed)
    df = df.copy()

    label_cols = [f"labels_{i}" for i in range(5)]

    # paper uses labels_1 as the primary label w.l.o.g
    df["final_label"] = df["labels_1"]

    # Helper to gather all other labels for a row
    def other_votes(row):
        return [row[c] for c in label_cols if row[c] not in (None, np.nan)]

    # Replace “Other” if someone else labeled non-“Other”, else drop
    mask_other = df["final_label"] == "Other"
    for idx in df[mask_other].index:
        votes = other_votes(df.loc[idx])
        non_other = [v for v in votes if v != "Other"]
        if non_other:
            df.at[idx, "final_label"] = non_other[0]
        else:
            df.drop(idx, inplace=True)

    # Reconcile “Safe” labels:
    mask_safe = df["final_label"] == "Safe"
    for idx in df[mask_safe].index:
        votes = other_votes(df.loc[idx])
        unsafe_votes = [v for v in votes if v != "Safe"]
        if unsafe_votes:
            # pick one of the unsafe categories at random
            df.at[idx, "final_label"] = np.random.choice(unsafe_votes)
        # else: all agree on Safe, keep “Safe”


    # Remove “Needs Caution”
    df = df[~df["final_label"].isin(["Needs Caution", 'Other'])]

    # Drop duplicates by prompt text (keep first occurrence)
    df = df.drop_duplicates(subset="text", keep="first")

    return df.drop(columns=label_cols)

def load_llm(model_name: str, hf_token: str = None, device: str = None):
    token = hf_token or os.getenv("HF_TOKEN")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=token).to(device)
    model.eval()
    return tokenizer, model, device

aegis_test_df = preprocess_aegis(aegis_test_df)
aegis_test_df = aegis_test_df[aegis_test_df['text_type'] == 'user_message']
# aegis_test_df = preprocess_aegis(aegis_test_df)

tokenizer, model, device = load_llm("meta-llama/Llama-3.2-3B-Instruct")


negatives = []
positives = []

premature_termination_count = 0
timed_out_count = 0
# Loop over safe prompts
for prompt in aegis_test_df[aegis_test_df['final_label'] == 'Safe']['text'].tolist()[:50]:
    print("Safe prompt:")
    print(prompt)
    planner = GuardRailPDDLStream(
        start_text=prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        nucleus_p=0.9,
        temperature=0.55,
        harm_detector_path="harm_detector/models/binary_harm_detector.dill",
        max_length=30,
    )
    solution = planner.solve(max_time=60)
    if solution.plan is None:
        timed_out_count += 1
        print('Timed out')
        negatives.append((prompt, True))
    else:
        print(solution.plan[-1][0])
        found_safe = 'mark-found-no-harm' in solution.plan[-1][0]
        if found_safe:
            if len(solution.plan) < 30:
                premature_termination_count += 1
        print(found_safe)
        negatives.append((prompt, found_safe))
    del planner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# Loop over harmful prompts
stop_iter = []
for prompt in aegis_test_df[aegis_test_df['final_label'] != 'Safe']['text'].tolist()[:50]:
    print("Harmful prompt:")
    print(prompt)
    planner = GuardRailPDDLStream(
        start_text=prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        nucleus_p=0.9,
        temperature=0.55,
        harm_detector_path="harm_detector/models/binary_harm_detector.dill",
        max_length=30,
    )
    solution = planner.solve(max_time=60)
    if solution.plan is None:
        timed_out_count += 1
        print('Timed out')
        positives.append((prompt, False))
    else:
        found_harm = 'mark-found-harm' in solution.plan[-1][0]
        positives.append((prompt, found_harm))
        if not found_harm:
            if len(solution.plan) < 30:
                premature_termination_count += 1
        print(found_harm)
        stop_iter.append(len(solution.plan))
    del planner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# Calculate metrics
true_positives = sum(1 for _, found in positives if found)
true_negatives = sum(1 for _, found in negatives if not found)
false_positives = sum(1 for _, found in positives if not found)
false_negatives = sum(1 for _, found in negatives if found)

precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Accuracy: {accuracy:.4f}")

avg_iters = np.mean(stop_iter) if stop_iter else 0
print(f"Average iterations for harmful prompts: {avg_iters:.2f}")
print(f"Premature termination count: {premature_termination_count}")
print(f"Timed out count: {timed_out_count}")

os.makedirs('results', exist_ok=True)
with open('results/grd_benchmark_results.txt', 'w') as f:
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Average iterations for harmful prompts: {avg_iters:.2f}\n")
    f.write(f"Premature termination count: {premature_termination_count}\n")
    f.write(f"Timed out count: {timed_out_count}\n")
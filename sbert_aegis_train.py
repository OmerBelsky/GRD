#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a Sentence-BERT (DistilBERT backbone) on the AEGIS dataset using a triplet loss,
then freeze it and train a shallow multi-class classifier on the embeddings.

Matches the recipe summarized from Zheng et al. (AEGIS paper):
- Base: distilbert-base-uncased
- Pooling: mean
- Loss: Batch-hard triplet loss with soft margin (margin=0.0)
- Batch size: 16
- Epochs: 10
- No projection / normalization layers
- Two-stage: embeddings -> shallow NN classifier
"""

import os
import random
import argparse
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer, models, losses, InputExample, evaluation
from torch.utils.data import DataLoader

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, average_precision_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# -----------------------
# Utils
# -----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_aegis_parquet(split: str) -> pd.DataFrame:
    """
    Load AEGIS split from HuggingFace parquet (requires fsspec>=2023).
    split: 'train' or 'test'
    """
    files = {
        "train": "Content Moderation Extracted Annotations 02.08.24_train_release_0418_v1.parquet",
        "test":  "Content Moderation Extracted Annotations 02.08.24_test_release_0418_v1.parquet",
    }
    path = f"hf://datasets/nvidia/Aegis-AI-Content-Safety-Dataset-1.0/{files[split]}"
    return pd.read_parquet(path)


def preprocess_aegis(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Reconcile multiple annotator labels into a single 'final_label' per row,
    following the AEGIS scheme described by Zheng et al.

    Steps:
      1) Start from labels_1 as the primary label.
      2) If it is 'Other' and any other label is non-'Other', replace with that non-'Other'; else drop row.
      3) If it is 'Safe' and any other label is unsafe (not 'Safe'), replace 'Safe' with one of those unsafe labels at random.
      4) Remove rows with 'Needs Caution' or 'Other'.
      5) Drop duplicate texts.
    """
    rng = np.random.default_rng(seed)
    df = df.copy()

    label_cols = [f"labels_{i}" for i in range(5)]

    # (0) some rows may have comma-separated multi-labels - drop those
    for col in label_cols:
        df = df[~df[col].fillna('').str.contains(", ")]

    df["final_label"] = df["labels_1"]

    def other_votes(row):
        return [row[c] for c in label_cols if pd.notna(row[c])]

    # Handle "Other"
    mask_other = df["final_label"] == "Other"
    for idx in df[mask_other].index:
        votes = other_votes(df.loc[idx])
        non_other = [v for v in votes if v != "Other"]
        if non_other:
            df.at[idx, "final_label"] = non_other[0]
        else:
            df.drop(idx, inplace=True)

    # Handle "Safe"
    mask_safe = df["final_label"] == "Safe"
    for idx in df[mask_safe].index:
        votes = other_votes(df.loc[idx])
        unsafe_votes = [v for v in votes if v != "Safe"]
        if unsafe_votes:
            df.at[idx, "final_label"] = rng.choice(unsafe_votes)

    # Remove Needs Caution / Other
    df = df[~df["final_label"].isin(["Needs Caution", "Other"])]

    # Drop duplicates by text
    df = df.drop_duplicates(subset="text", keep="first")

    # Keep only necessary columns
    keep_cols = ["id", "text", "final_label"]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].reset_index(drop=True)


def build_sbert(max_seq_len: int = 512) -> SentenceTransformer:
    """
    DistilBERT backbone + mean pooling (CLS/max disabled).
    """
    transformer = models.Transformer("distilbert-base-uncased", max_seq_length=max_seq_len)
    pooling = models.Pooling(
        word_embedding_dimension=transformer.get_word_embedding_dimension(),
        pooling_mode_mean_tokens=True,
        pooling_mode_cls_token=False,
        pooling_mode_max_tokens=False,
    )
    model = SentenceTransformer(modules=[transformer, pooling])
    return model


def make_dataloader(texts: List[str], labels: List[int], batch_size: int = 16) -> DataLoader:
    """
    Create a DataLoader of InputExample where .label is an int class id.
    We'll use BatchHardTripletLoss which mines triplets inside each batch
    based on these labels.
    """
    examples = [InputExample(texts=[t], label=int(y)) for t, y in zip(texts, labels)]
    loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=True,
                        collate_fn=SentenceTransformer.smart_batching_collate)
    return loader


def train_sbert(model: SentenceTransformer,
                train_loader: DataLoader,
                epochs: int = 10,
                lr: float = 2e-5,
                warmup_frac: float = 0.1,
                output_dir: str = "output/sbert_aegis") -> str:
    """
    Train with Batch-Hard Triplet Loss (soft margin when margin=0.0).
    """
    os.makedirs(output_dir, exist_ok=True)

    train_loss = losses.BatchHardTripletLoss(
        model=model,
        distance_metric=losses.SiameseDistanceMetric.COSINE,
        margin=0.0  # soft margin
    )

    warmup_steps = int(len(train_loader) * epochs * warmup_frac)

    model.fit(
        train_objectives=[(train_loader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        optimizer_params={'lr': lr},
        output_path=output_dir,
        show_progress_bar=True,
    )
    return output_dir


def embed(model: SentenceTransformer, texts: List[str], batch_size: int = 256) -> np.ndarray:
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=False)


def train_classifier(X: np.ndarray, y: np.ndarray, hidden: int = 256, seed: int = 42) -> MLPClassifier:
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), activation='relu', solver='adam',
                        alpha=1e-4, max_iter=100, random_state=seed, verbose=False)
    clf.fit(X, y)
    return clf


def evaluate(y_true, y_pred, probs, label_encoder: LabelEncoder):
    print("\n=== Multiclass metrics ===")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_, digits=4))

    # Macro F1 (multiclass)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    print(f"Macro F1: {macro_f1:.4f}")

    # One-vs-rest AUPRC macro
    # Build a score matrix with columns aligned to label order
    Y_true_ovr = np.zeros_like(probs)
    for idx, label in enumerate(y_true):
        Y_true_ovr[idx, label] = 1
    auprc_per_class = []
    for c in range(probs.shape[1]):
        try:
            auprc_per_class.append(average_precision_score(Y_true_ovr[:, c], probs[:, c]))
        except ValueError:
            pass
    if auprc_per_class:
        print(f"Macro AUPRC (OvR): {np.mean(auprc_per_class):.4f}")

    # Collapsed Safe-vs-Unsafe
    if "Safe" in label_encoder.classes_:
        safe_id = np.where(label_encoder.classes_ == "Safe")[0][0]
        y_true_bin = (y_true == safe_id).astype(int)
        y_pred_bin = (y_pred == safe_id).astype(int)
        # score for "Safe" vs not
        probs_safe = probs[:, safe_id] if probs.shape[1] > safe_id else (y_pred_bin.astype(float))
        print("\n=== Collapsed Safe vs Unsafe ===")
        print("F1 (binary, Safe positive):", f1_score(y_true_bin, y_pred_bin))
        try:
            print("AUPRC (Safe as positive):", average_precision_score(y_true_bin, probs_safe))
        except ValueError:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_frac", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="output/sbert_aegis")
    args = parser.parse_args()

    set_seed(args.seed)

    print("Loading AEGIS train/test parquet...")
    train_df = load_aegis_parquet("train")
    test_df = load_aegis_parquet("test")

    print("Preprocessing...")
    train_df = preprocess_aegis(train_df, seed=args.seed)
    test_df  = preprocess_aegis(test_df,  seed=args.seed)

    # Label encode final_label over union of train+test to keep indices aligned
    le = LabelEncoder()
    le.fit(pd.concat([train_df["final_label"], test_df["final_label"]], axis=0).values)

    y_train = le.transform(train_df["final_label"].values)
    y_test  = le.transform(test_df["final_label"].values)

    # -----------------------
    # SBERT fine-tuning
    # -----------------------
    print("Building Sentence-BERT...")
    model = build_sbert(max_seq_len=args.max_seq_len)

    print("Preparing DataLoader...")
    train_loader = make_dataloader(train_df["text"].tolist(), y_train.tolist(), batch_size=args.batch_size)

    print("Training SBERT (triplet, batch-hard soft-margin)...")
    save_dir = train_sbert(model, train_loader, epochs=args.epochs, lr=args.lr,
                           warmup_frac=args.warmup_frac, output_dir=args.output_dir)

    print(f"Model saved to: {save_dir}")

    # Reload the best model just in case
    model = SentenceTransformer(save_dir)

    # -----------------------
    # Embeddings & classifier
    # -----------------------
    print("Encoding embeddings...")
    X_train = embed(model, train_df["text"].tolist())
    X_test  = embed(model, test_df["text"].tolist())

    print("Training shallow NN classifier on embeddings...")
    clf = train_classifier(X_train, y_train, hidden=256, seed=args.seed)

    print("Evaluating...")
    y_pred = clf.predict(X_test)
    probs  = clf.predict_proba(X_test)
    evaluate(y_test, y_pred, probs, le)

    # Save artifacts
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(args.output_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(args.output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.output_dir, "y_test.npy"),  y_test)

    import joblib
    joblib.dump(clf, os.path.join(args.output_dir, "mlp_classifier.joblib"))
    with open(os.path.join(args.output_dir, "label_encoder.classes.txt"), "w", encoding="utf-8") as f:
        for c in le.classes_:
            f.write(str(c) + "\n")

    print("\nDone. Artifacts saved under:", args.output_dir)


if __name__ == "__main__":
    main()

import os
import torch
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics import classification_report, accuracy_score, f1_score
import pickle
from harm_detector import HarmDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load distilbert tokenizer and model
distilbert_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

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


aegis_train_df = preprocess_aegis(aegis_train_df)
aegis_test_df = preprocess_aegis(aegis_test_df)

clf = HarmDetector(bert_model=distilbert_model, tokenizer=distilbert_tokenizer, device=device)

y_train = aegis_train_df['final_label']
y_test = aegis_test_df['final_label']
clf.fit(aegis_train_df, y_train)
# Evaluate the model
y_pred = clf.predict(aegis_test_df)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("F1 Score (macro):", f1_score(y_test, y_pred, average='macro'))

# Evaluate on training set
y_train_pred = clf.predict(aegis_train_df)
print(classification_report(y_train, y_train_pred))
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("F1 Score:", f1_score(y_train, y_train_pred, average='weighted'))
print("F1 Score (macro):", f1_score(y_train, y_train_pred, average='macro'))

# Save the SVM
if not os.path.exists("models"):
    os.makedirs("models")
    with open("models/harm_detector.pkl", "wb") as f:
        pickle.dump(clf, f)
"""Helper functions shared across the pipeline.

Main use here: build a stratified train / validation / test split.

Reads  : data/labeled/labeled.csv
Writes : data/splits/{train,val,test}.csv

Split = 70 / 15 / 15, stratified by label so every category keeps the same
proportion in each subset. The test set stays untouched until final evaluation.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

LABELED = "data/labeled/labeled.csv"
SPLIT_DIR = "data/splits"
SEED = 42

CATEGORIES = ["intern", "junior", "senior", "lead", "template", "low_value"]
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def make_splits(df: pd.DataFrame, seed: int = SEED):
    """Return (train, val, test) DataFrames stratified by label."""
    # first carve out 15% test
    train_val, test = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=seed)
    # then 15/85 of the remainder -> ~15% overall validation
    train, val = train_test_split(
        train_val, test_size=0.1765, stratify=train_val["label"],
        random_state=seed)
    return train, val, test


def main():
    if not os.path.exists(LABELED):
        raise SystemExit(f"Missing {LABELED}. Run llm_labeling.py first.")
    df = pd.read_csv(LABELED)
    df["label_id"] = df["label"].map(LABEL2ID)

    train, val, test = make_splits(df)

    os.makedirs(SPLIT_DIR, exist_ok=True)
    train.to_csv(f"{SPLIT_DIR}/train.csv", index=False)
    val.to_csv(f"{SPLIT_DIR}/val.csv", index=False)
    test.to_csv(f"{SPLIT_DIR}/test.csv", index=False)

    print(f"train={len(train)}  val={len(val)}  test={len(test)}  "
          f"(total={len(df)})\n")
    dist = pd.DataFrame({
        "train": train["label"].value_counts(),
        "val": val["label"].value_counts(),
        "test": test["label"].value_counts(),
    }).fillna(0).astype(int)
    print("Label distribution per split:")
    print(dist)


if __name__ == "__main__":
    main()

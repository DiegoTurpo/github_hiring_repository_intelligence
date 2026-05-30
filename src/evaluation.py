"""Metrics, validation and error analysis.

Reads  : output/predictions/test_predictions.csv  (produced in Colab by train.py)
Writes : output/metrics/metrics.json
         output/tables/classification_report.csv
         output/tables/most_confused_pairs.csv
         output/tables/misclassified_examples.csv

No torch needed: works entirely from the exported predictions.
"""

import os
import json
from collections import Counter

import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

PRED = "output/predictions/test_predictions.csv"
CATEGORIES = ["intern", "junior", "senior", "lead", "template", "low_value"]


def main():
    if not os.path.exists(PRED):
        raise SystemExit(f"Missing {PRED}. Run training in Colab first.")
    df = pd.read_csv(PRED)
    y_true, y_pred = df["true_label"], df["pred_label"]

    os.makedirs("output/metrics", exist_ok=True)
    os.makedirs("output/tables", exist_ok=True)

    # 1) headline metrics
    metrics = {
        "n_test": len(df),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0), 4),
        "precision_macro": round(precision_score(y_true, y_pred,
                                 average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, average="macro",
                              zero_division=0), 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted",
                             zero_division=0), 4),
    }
    with open("output/metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Headline metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 2) per-class report
    report = classification_report(
        y_true, y_pred, labels=CATEGORIES, zero_division=0, output_dict=True)
    rep_df = pd.DataFrame(report).T.round(3)
    rep_df.to_csv("output/tables/classification_report.csv")
    print("\nPer-class report:")
    print(rep_df.loc[CATEGORIES][["precision", "recall", "f1-score", "support"]])

    # 3) most confused pairs (true != pred)
    errors = df[y_true != y_pred]
    pairs = Counter(zip(errors["true_label"], errors["pred_label"]))
    confused = pd.DataFrame(
        [{"true": t, "predicted": p, "count": c}
         for (t, p), c in pairs.most_common()])
    confused.to_csv("output/tables/most_confused_pairs.csv", index=False)
    print(f"\nTotal errors: {len(errors)}/{len(df)}")
    if len(confused):
        print("Most confused pairs (true -> predicted):")
        print(confused.head(6).to_string(index=False))

    # 4) misclassified examples for qualitative inspection
    errors[["full_name", "true_label", "pred_label", "pred_confidence"]].to_csv(
        "output/tables/misclassified_examples.csv", index=False)
    print(f"\nSaved metrics, report and error tables to output/")


if __name__ == "__main__":
    main()

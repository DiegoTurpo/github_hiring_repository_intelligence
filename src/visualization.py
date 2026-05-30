"""Graphs and analytical visualizations.

Reads  : output/predictions/test_predictions.csv
         data/labeled/labeled.csv
         data/processed/repos_clean.csv
Writes : output/figures/*.png

No torch needed: works from exported predictions and the labeled dataset.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

CATEGORIES = ["intern", "junior", "senior", "lead", "template", "low_value"]
FIG_DIR = "output/figures"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{name}", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {FIG_DIR}/{name}")


def plot_confusion_matrix(df):
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=CATEGORIES)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CATEGORIES, yticklabels=CATEGORIES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (test set)")
    save(fig, "confusion_matrix.png")


def plot_per_class_f1(df):
    f1 = f1_score(df["true_label"], df["pred_label"],
                  labels=CATEGORIES, average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=CATEGORIES, y=f1, ax=ax, hue=CATEGORIES, legend=False,
                palette="viridis")
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 (test set)")
    for i, v in enumerate(f1):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    save(fig, "per_class_f1.png")


def plot_label_distribution(labeled):
    counts = labeled["label"].value_counts().reindex(CATEGORIES).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index,
                legend=False, palette="mako")
    ax.set_ylabel("Number of repositories")
    ax.set_title("Weak-label distribution (full dataset)")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 1, int(v), ha="center")
    save(fig, "label_distribution.png")


def plot_signal_by_category(labeled, clean):
    """How a key signal (stars) varies across categories."""
    merged = labeled.merge(clean[["full_name", "stars", "contributors",
                                  "has_ci"]], on="full_name", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(data=merged, x="label", y="stars", order=CATEGORIES,
                ax=axes[0], hue="label", legend=False, palette="rocket")
    axes[0].set_yscale("log")
    axes[0].set_title("Stars by category (log scale)")
    axes[0].tick_params(axis="x", rotation=30)

    ci = merged.groupby("label")["has_ci"].mean().reindex(CATEGORIES)
    sns.barplot(x=ci.index, y=ci.values, ax=axes[1], hue=ci.index,
                legend=False, palette="rocket")
    axes[1].set_title("Share with CI/CD by category")
    axes[1].set_ylabel("fraction with CI")
    axes[1].tick_params(axis="x", rotation=30)
    save(fig, "signals_by_category.png")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    preds = pd.read_csv("output/predictions/test_predictions.csv")
    labeled = pd.read_csv("data/labeled/labeled.csv")
    clean = pd.read_csv("data/processed/repos_clean.csv")

    print("Generating figures:")
    plot_confusion_matrix(preds)
    plot_per_class_f1(preds)
    plot_label_distribution(labeled)
    plot_signal_by_category(labeled, clean)
    print("Done.")


if __name__ == "__main__":
    main()

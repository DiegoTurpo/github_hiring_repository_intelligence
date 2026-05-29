"""Cleaning and transformations of raw repository data.

Reads  : data/raw/repos.csv
Writes : data/processed/repos_clean.csv

Steps
-----
1. Drop exact duplicates and rows without a name.
2. Coerce numeric signals, fill missing values with 0.
3. Derive a few interpretable features used later for analysis and for the
   textual representation (commits per contributor, README presence, activity
   recency bucket, etc.).
"""

import os
import pandas as pd

RAW = "data/raw/repos.csv"
OUT = "data/processed/repos_clean.csv"

NUMERIC_COLS = [
    "stars", "forks", "open_issues", "contributors", "commits", "releases",
    "has_ci", "readme_length", "n_topics", "has_license", "age_days",
    "days_since_push",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset="full_name").copy()
    df = df[df["full_name"].notna()]

    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["language"] = df["language"].fillna("unknown")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    # avoid division by zero
    safe_contributors = df["contributors"].replace(0, 1)
    df["commits_per_contributor"] = (df["commits"] / safe_contributors).round(1)
    df["has_readme"] = (df["readme_length"] > 0).astype(int)
    df["age_years"] = (df["age_days"] / 365).round(1)

    # recency bucket for activity
    def recency(days):
        if days <= 30:
            return "active"
        if days <= 365:
            return "recent"
        return "stale"

    df["activity"] = df["days_since_push"].apply(recency)
    return df


def main():
    if not os.path.exists(RAW):
        raise SystemExit(f"Missing {RAW}. Run github_collector.py first.")
    df = pd.read_csv(RAW)
    print(f"Loaded {len(df)} raw repos")

    df = clean(df)
    df = add_derived_features(df)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} cleaned repos -> {OUT}")
    print(df[["full_name", "stars", "contributors", "commits_per_contributor",
              "activity"]].head().to_string(index=False))


if __name__ == "__main__":
    main()

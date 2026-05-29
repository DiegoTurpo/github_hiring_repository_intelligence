"""Repository summary / textual representation generation.

Reads  : data/processed/repos_clean.csv
Writes : data/processed/repos_text.csv   (adds a 'text' column)

Why a textual representation?
-----------------------------
Both the LLM (weak labeling) and the BERT model consume text. We turn the
numeric signals into a compact natural-language description that mixes:
  - qualitative descriptors (e.g. "very popular", "single contributor"), and
  - the raw numbers,
so the models get both human-readable cues and exact values. Keeping the same
representation for labeling and training avoids train/label mismatch.
"""

import os
import pandas as pd

IN = "data/processed/repos_clean.csv"
OUT = "data/processed/repos_text.csv"


def bucket(value, thresholds, labels):
    """Map a value to a qualitative label given ascending thresholds."""
    for t, lab in zip(thresholds, labels):
        if value <= t:
            return lab
    return labels[-1]


def describe_popularity(stars):
    return bucket(stars, [3, 100, 1000, 10000],
                  ["almost no", "low", "moderate", "high", "very high"])


def describe_team(contributors):
    if contributors <= 1:
        return "a single contributor"
    if contributors <= 5:
        return f"a small team of {contributors} contributors"
    if contributors <= 50:
        return f"a team of {contributors} contributors"
    return f"a large community of {contributors} contributors"


def build_text(row) -> str:
    pop = describe_popularity(row["stars"])
    team = describe_team(int(row["contributors"]))
    ci = "has CI/CD workflows" if row["has_ci"] else "lacks CI/CD workflows"
    lic = "licensed" if row["has_license"] else "no license"
    readme = (f"a README of {int(row['readme_length'])} characters"
              if row["has_readme"] else "no README")
    topics = (f"{int(row['n_topics'])} topics" if row["n_topics"] > 0
              else "no topics")

    return (
        f"Repository in {row['language']} with {pop} popularity "
        f"({int(row['stars'])} stars, {int(row['forks'])} forks). "
        f"Maintained by {team}, {int(row['commits'])} total commits "
        f"({row['commits_per_contributor']} per contributor). "
        f"{int(row['releases'])} releases, {int(row['open_issues'])} open issues. "
        f"It {ci}, has {readme}, {topics}, and is {lic}. "
        f"The project is {row['age_years']} years old and currently "
        f"{row['activity']} (last push {int(row['days_since_push'])} days ago)."
    )


def main():
    if not os.path.exists(IN):
        raise SystemExit(f"Missing {IN}. Run preprocessing.py first.")
    df = pd.read_csv(IN)
    df["text"] = df.apply(build_text, axis=1)

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved {len(df)} repos with text -> {OUT}\n")
    print("Example representations:\n")
    for _, r in df.head(3).iterrows():
        print(f"- {r['full_name']}")
        print(f"  {r['text']}\n")


if __name__ == "__main__":
    main()

"""Weak labeling of repositories using an LLM (Groq / Llama 3.3 70B).

Reads  : data/processed/repos_text.csv
Writes : data/labeled/labeled.csv   (resumable, written incrementally)

The LLM acts as the *initial annotator*. It receives the textual repository
representation and assigns ONE of six engineering-maturity categories, plus a
confidence score and a short justification. These are WEAK labels: noisy,
generated without seeing the source code, only repository metadata.

Provider choice
---------------
Groq free tier (llama-3.1-8b-instant) -> 500K tokens/day free, enough to label
the whole dataset in one run at no cost. (The larger 70B model has only ~100K
tokens/day, which is exhausted before finishing 486 repos.)
"""

import os
import json
import time

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "llama-3.1-8b-instant"
IN = "data/processed/repos_text.csv"
OUT = "data/labeled/labeled.csv"
SLEEP_SECONDS = 2.2  # ~27 requests/min, safely under the 30 RPM free-tier limit

CATEGORIES = ["intern", "junior", "senior", "lead", "template", "low_value"]

SYSTEM_INSTRUCTION = """Estimate the engineering MATURITY a GitHub repository
reflects (judge the repo, not the developer). Pick exactly one category:
- intern: learning/tutorial/exercise; minimal structure, no tests/CI, tiny
  README, single author, very few commits.
- junior: real but simple; limited practices, little/no CI/tests, modest docs.
- senior: solid engineering; CI/CD, tests, good docs, releases, several
  contributors, sustained activity.
- lead: large, mature, multi-contributor, many releases, broad community.
- template: boilerplate/starter/scaffold/replica; value is being reusable.
- low_value: empty, abandoned, trivial, no clear purpose.
Reply in strict JSON: {"category","confidence"(0-1),"reason"(short)}."""


def build_client():
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "your_groq_key_here":
        raise SystemExit("Set GROQ_API_KEY in your .env file first.")
    return Groq(api_key=key)


def label_one(client, text):
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": f"Repository description:\n{text}"},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    data = json.loads(resp.choices[0].message.content)
    cat = str(data.get("category", "")).strip().lower()
    if cat not in CATEGORIES:
        cat = "low_value"  # fallback for unexpected output
    return cat, float(data.get("confidence", 0.0)), str(data.get("reason", ""))[:300]


def load_done():
    """Return set of already-labeled full_names (resume support)."""
    if os.path.exists(OUT):
        done = pd.read_csv(OUT)
        return set(done["full_name"]), done
    return set(), pd.DataFrame()


def main():
    df = pd.read_csv(IN)
    client = build_client()
    done_names, done_df = load_done()

    todo = df[~df["full_name"].isin(done_names)]
    print(f"{len(done_names)} already labeled, {len(todo)} to go (model={MODEL})")

    os.makedirs("data/labeled", exist_ok=True)
    rows = done_df.to_dict("records") if len(done_df) else []

    for i, r in enumerate(todo.itertuples(index=False), 1):
        try:
            cat, conf, reason = label_one(client, r.text)
        except Exception as e:
            print(f"  [{i}] error on {r.full_name}: {str(e)[:200]}")
            time.sleep(SLEEP_SECONDS)
            continue

        rows.append({"full_name": r.full_name, "text": r.text,
                     "label": cat, "confidence": conf, "reason": reason})
        pd.DataFrame(rows).to_csv(OUT, index=False)  # incremental save
        if i % 10 == 0 or i == len(todo):
            print(f"  [{i}/{len(todo)}] {r.full_name} -> {cat} ({conf:.2f})")
        time.sleep(SLEEP_SECONDS)

    final = pd.DataFrame(rows)
    print(f"\nSaved {len(final)} labels -> {OUT}")
    print("\nLabel distribution:")
    print(final["label"].value_counts())


if __name__ == "__main__":
    main()

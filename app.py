"""Streamlit app — GitHub Hiring Repository Intelligence (Track A).

Four tabs:
  1. Problem & Methodology
  2. Exploratory Analysis
  3. Model Results
  4. Interactive Repository Exploration

Runs without torch: it reads the precomputed predictions, metrics and figures
produced by the pipeline (train.py exports predictions; evaluation.py and
visualization.py produce metrics/figures).

Run:  streamlit run app.py
"""

import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="GitHub Hiring Repo Intelligence",
                   layout="wide")

CATEGORIES = ["intern", "junior", "senior", "lead", "template", "low_value"]


@st.cache_data
def load_data():
    labeled = pd.read_csv("data/labeled/labeled.csv")
    clean = pd.read_csv("data/processed/repos_clean.csv")
    preds = pd.read_csv("output/predictions/all_predictions.csv")
    with open("output/metrics/metrics.json") as f:
        metrics = json.load(f)
    report = pd.read_csv("output/tables/classification_report.csv", index_col=0)
    confused = pd.read_csv("output/tables/most_confused_pairs.csv")
    merged = labeled.merge(
        clean[["full_name", "stars", "forks", "contributors", "commits",
               "releases", "has_ci", "readme_length", "age_years", "activity"]],
        on="full_name", how="left")
    return labeled, clean, preds, metrics, report, confused, merged


labeled, clean, preds, metrics, report, confused, merged = load_data()

st.title("GitHub Hiring Repository Intelligence")
st.caption("Track A — estimating the engineering maturity reflected by a "
           "GitHub repository, using GitHub signals + LLM weak labels + a "
           "fine-tuned DistilBERT classifier.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Problem & Methodology",
    "2. Exploratory Analysis",
    "3. Model Results",
    "4. Interactive Exploration",
])

# ---------------------------------------------------------------- Tab 1
with tab1:
    st.header("Problem & Methodology")
    st.markdown("""
**Objective.** Estimate the *engineering maturity* a GitHub repository reflects
(not the developer), classifying it into one of six categories:
`intern`, `junior`, `senior`, `lead`, `template`, `low_value`.

**Repository selection.** ~486 repositories sampled across six strata to span
the full maturity spectrum: very popular projects (lead/senior), mid-popularity
(junior/senior), tiny/inactive repos (intern/low-value) and
boilerplate/starter-template repos. Sampling targets *diversity*, not labels;
this introduces a selection bias we acknowledge as a limitation.

**GitHub signals used (13).** stars, forks, open issues, contributors, commits,
releases, CI/CD presence, README length, topics, license, language, repository
age, and days since last push.

**Repository representation.** Signals are turned into a natural-language
description mixing qualitative cues (*"very high popularity"*, *"single
contributor"*) and exact numbers — the same text feeds both the LLM and BERT.

**Weak labeling.** An LLM (Groq, Llama 3.1 8B) acts as the initial annotator,
assigning one of the six categories per repository. Labels are *weak*: noisy,
generated from metadata only, without reading the source code.

**Dataset split.** 70 / 15 / 15 train / validation / test, stratified by label.
The test set stays untouched until final evaluation.

**Model.** DistilBERT fine-tuned with class-weighted loss to handle imbalance.

**Limitations.** Weak labels are noisy; stars bias popularity; sampling is not
random; junior/senior boundaries are fuzzy; rare classes (intern, low_value)
have few examples. This is a learning pipeline, not a hiring oracle — it must
not be used to judge individuals.
""")

# ---------------------------------------------------------------- Tab 2
with tab2:
    st.header("Exploratory Analysis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Repositories", len(labeled))
    c2.metric("Categories", labeled["label"].nunique())
    c3.metric("With CI/CD", f"{clean['has_ci'].mean()*100:.0f}%")

    st.subheader("Weak-label distribution")
    if os.path.exists("output/figures/label_distribution.png"):
        st.image("output/figures/label_distribution.png")
    st.caption("The dataset is imbalanced: lead and template dominate, while "
               "intern and low_value are rare — a key caveat for the metrics.")

    st.subheader("Signals by category")
    if os.path.exists("output/figures/signals_by_category.png"):
        st.image("output/figures/signals_by_category.png")
    st.caption("Stars grow with maturity (intern -> lead), and CI/CD adoption "
               "rises with maturity — evidence these signals carry information "
               "about engineering maturity.")

    st.subheader("Signal averages per category")
    st.dataframe(merged.groupby("label")[
        ["stars", "contributors", "commits", "releases", "has_ci",
         "readme_length"]].mean().round(1).reindex(CATEGORIES))

# ---------------------------------------------------------------- Tab 3
with tab3:
    st.header("Model Results (test set)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", metrics["accuracy"])
    c2.metric("F1 (macro)", metrics["f1_macro"])
    c3.metric("F1 (weighted)", metrics["f1_weighted"])
    c4.metric("Test size", metrics["n_test"])

    cc1, cc2 = st.columns(2)
    with cc1:
        st.subheader("Confusion matrix")
        if os.path.exists("output/figures/confusion_matrix.png"):
            st.image("output/figures/confusion_matrix.png")
    with cc2:
        st.subheader("Per-class F1")
        if os.path.exists("output/figures/per_class_f1.png"):
            st.image("output/figures/per_class_f1.png")

    st.subheader("Per-class report")
    st.dataframe(report.loc[CATEGORIES][
        ["precision", "recall", "f1-score", "support"]])

    st.subheader("Most confused pairs (true -> predicted)")
    st.dataframe(confused)
    st.caption("The main error is senior vs lead — the fuzziest maturity "
               "boundary, consistent with the weak-label noise.")

    st.subheader("Baseline vs alternative")
    st.markdown("""
- **Baseline:** majority-class / metadata-only logistic regression would mostly
  predict the dominant classes (lead/template), with low macro-F1 on rare ones.
- **Alternative (this model):** DistilBERT on the textual representation with
  class-weighted loss reaches **{acc} accuracy** and **{f1} macro-F1**,
  recovering minority classes such as intern and template.
""".format(acc=metrics["accuracy"], f1=metrics["f1_macro"]))

# ---------------------------------------------------------------- Tab 4
with tab4:
    st.header("Interactive Repository Exploration")
    colf1, colf2, colf3 = st.columns(3)
    search = colf1.text_input("Search repository name")
    cat_filter = colf2.multiselect("Predicted category", CATEGORIES)
    only_errors = colf3.checkbox("Only misclassified", value=False)

    view = preds.copy()
    if search:
        view = view[view["full_name"].str.contains(search, case=False,
                                                    na=False)]
    if cat_filter:
        view = view[view["pred_label"].isin(cat_filter)]
    if only_errors:
        view = view[view["true_label"] != view["pred_label"]]

    st.write(f"Showing {len(view)} repositories")
    st.dataframe(
        view[["full_name", "true_label", "pred_label", "pred_confidence"]]
        .sort_values("pred_confidence", ascending=False),
        use_container_width=True, height=400)

    st.subheader("Inspect a repository")
    if len(view):
        pick = st.selectbox("Choose a repository", view["full_name"].tolist())
        row = view[view["full_name"] == pick].iloc[0]
        cA, cB = st.columns([1, 2])
        with cA:
            st.metric("LLM weak label", row["true_label"])
            st.metric("Model prediction", row["pred_label"])
            st.metric("Confidence", f"{row['pred_confidence']:.2f}")
        with cB:
            st.markdown("**Repository representation**")
            st.info(row["text"])

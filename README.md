# GitHub Hiring Repository Intelligence

**Track A — Hiring-Oriented Repository Intelligence**

A weak-supervision NLP pipeline that analyzes GitHub repositories and estimates the
**engineering maturity** they reflect — `intern`, `junior`, `senior`, `lead`,
`template`, `low_value`. The unit of analysis is the **repository**, not the
developer: we estimate the engineering complexity and maturity *reflected by the
project itself*, never to judge or rank individuals.

The pipeline follows the classic **weak-supervision** recipe: an LLM generates noisy
("weak") labels from repository metadata, and a fine-tuned DistilBERT classifier
learns to generalize from those labels.

---

## TL;DR — Results

| Metric | Value |
|---|---|
| Test set size | 73 repos (held-out 15%) |
| Accuracy | **0.781** |
| F1 macro | **0.615** |
| F1 weighted | **0.768** |
| Precision macro / Recall macro | 0.590 / 0.673 |

Best per-class F1: `template` (0.92), `lead` (0.84). Weakest: `low_value` (0.00,
only 2 test examples). The dominant error is **senior ↔ lead** — the fuzziest
maturity boundary — consistent with weak-label noise.

---

## Pipeline

| Step | Script | Input → Output |
|---|---|---|
| 1. GitHub data collection | `src/github_collector.py` | GitHub API → `data/raw/repos.csv` |
| 2. Preprocessing | `src/preprocessing.py` | raw → `data/processed/repos_clean.csv` |
| 3. Repository representation | `src/summarization.py` | clean → `data/processed/repos_text.csv` |
| 4. LLM weak labeling | `src/llm_labeling.py` | text → `data/labeled/labeled.csv` |
| 5. Train/val/test split | `src/utils.py` | labeled → `data/splits/{train,val,test}.csv` |
| 6. BERT fine-tuning | `src/train.py` (Colab GPU) | splits → model + `output/predictions/*.csv` |
| 7. Evaluation | `src/evaluation.py` | predictions → `output/metrics/`, `output/tables/` |
| 8. Visualization | `src/visualization.py` | predictions → `output/figures/*.png` |
| 9. Dashboard | `app.py` | all artifacts → Streamlit app (4 tabs) |

**Key design choice:** `torch` is used **only** during training (Colab T4 GPU).
`train.py` exports predictions to CSV, so evaluation, visualization and the
Streamlit app all run **locally without torch** (the local environment is Python
3.14, for which no torch wheels exist yet).

---

## Required questions

**1. What problem are you solving, and why does it matter?**
Estimating the engineering maturity reflected by a GitHub repository. For technical
hiring and portfolio review, recruiters face thousands of repos and no scalable,
consistent signal of project maturity. An automated, explainable estimate helps
triage portfolios — *as a first-pass signal, never as an automated hiring decision*.

**2. What is the unit of analysis?**
The **repository**, characterized by 13 repository-level GitHub signals. We never
model the person.

**3. How did you select the repositories?**
~486 repositories were sampled via **stratified search** across six strata designed
to span the full maturity spectrum: very popular projects (lead/senior),
mid-popularity (junior/senior), tiny/inactive repos (intern/low_value), and
boilerplate/starter-template repos. Sampling targets **diversity**, not labels — we
do not pre-label during sampling. This introduces a selection bias we acknowledge
as a limitation.

**4. What signals (features) did you extract? (13)**
stars, forks, open issues, contributors, commits, releases, CI/CD presence,
README length, topics, license, primary language, repository age, and days since
last push. Counts that the API paginates (contributors, commits) are recovered from
the `Link` header's `last` page.

**5. How did you turn metadata into model input (the representation)?**
`src/summarization.py` converts signals into a natural-language description that
mixes qualitative cues (*"very high popularity"*, *"single contributor"*) with exact
numbers, and includes the repo name (so `*-template`/`*-boilerplate` projects are
recognizable). The **same text** feeds both the LLM annotator and DistilBERT.

**6. What are the categories, and what do they mean?**
`intern` (learning/toy projects), `junior` (functional but simple), `senior`
(substantial, well-structured), `lead` (large, mature, widely adopted), `template`
(boilerplate/starter), `low_value` (abandoned/empty/near-zero signal).

**7. How did you generate labels (weak supervision)?**
An LLM — **Groq, `llama-3.1-8b-instant`** — acts as the initial annotator, assigning
one of the six categories per repo from its text representation (JSON mode,
`temperature=0.0`). Labels are **weak**: noisy, generated from metadata only, without
reading source code. Labeling is rate-limited and **resumable** (it skips repos
already labeled), so a 429 mid-run never loses progress.

**8. What is the label distribution? Is it balanced?**
Imbalanced. Full dataset (486): `lead` 181, `template` 133, `senior` 102,
`junior` 45, `intern` 15, `low_value` 10. `lead`/`template` dominate; `intern`/
`low_value` are rare — a key caveat for the metrics.

**9. How did you split the data?**
Stratified **70 / 15 / 15** (340 / 73 / 73), `seed=42`, via two `train_test_split`
calls. The **test set stays untouched** until final evaluation.

**10. What model did you use and why?**
`distilbert-base-uncased` fine-tuned for sequence classification (`MAX_LEN=256`,
5 epochs, batch 16, LR 2e-5). DistilBERT is lightweight enough for the Colab free
tier yet captures the textual nuance the qualitative cues encode. To counter
imbalance we use **class-weighted cross-entropy** (inverse class frequency) via a
`WeightedTrainer` subclass, so minority classes are not ignored.

**11. What are your results?**
Accuracy **0.781**, F1 macro **0.615**, F1 weighted **0.768** on the held-out test
set (n=73). Per-class F1: `template` 0.92, `lead` 0.84, `intern`/`junior` 0.67,
`senior` 0.59, `low_value` 0.00. See `output/figures/` and `output/tables/`.

**12. What is your baseline, and how does the model compare?**
Baseline: a majority-class / metadata-only predictor would mostly output the
dominant classes (`lead`/`template`) and score near-zero macro-F1 on rare ones. The
DistilBERT model on the textual representation reaches 0.781 accuracy / 0.615
macro-F1, **recovering minority classes** such as `intern` and `template` that a
majority baseline collapses.

**13. What are the limitations and risks?**
Weak labels are noisy (LLM never reads code); stars bias toward popularity;
sampling is not random (selection bias); `junior`/`senior` boundaries are fuzzy;
rare classes (`intern`, `low_value`) have very few examples (test F1 unreliable —
`low_value` has only 2 test rows). **Ethical risk:** this must **not** be used to
judge individuals or make automated hiring decisions — it is a learning pipeline and
a triage aid, not a hiring oracle.

---

## Track A — analytical questions

**A1. Which signals most distinguish maturity levels?**
Stars rise monotonically with maturity (`intern` → `lead`, visible on a log scale),
and **CI/CD adoption** climbs with maturity — mature projects automate testing.
Contributor count and commit volume also separate `lead`/`senior` from
`intern`/`low_value`. See `output/figures/signals_by_category.png`.

**A2. Where does the model get confused, and why?**
The top confusion is **senior → lead** (5 cases) and `lead → senior` (2). These
categories share high stars/activity and differ mostly in scale and adoption — a
genuinely fuzzy human boundary, so the weak labels themselves disagree there. See
`output/tables/most_confused_pairs.csv` and the confusion matrix.

**A3. How would this be used in a real hiring workflow?**
As a **first-pass triage signal**: rank/cluster a candidate's repos by reflected
maturity to surface their strongest work, flag boilerplate/low-value repos, and give
reviewers an explainable starting point (the text representation is shown alongside
the prediction). A human always makes the final call.

**A4. What would you improve with more time/budget?**
(1) Reduce label noise — multi-LLM agreement or a small human-validated gold set;
(2) gather more `intern`/`low_value` examples to fix rare-class metrics;
(3) add code-level signals (test coverage, lint, dependency health) beyond metadata;
(4) calibrate confidence and add an "abstain" option for low-confidence repos.

---

## Project structure

```
github_hiring_repository_intelligence/
├── app.py                      # Streamlit dashboard (4 tabs)
├── README.md
├── requirements.txt
├── src/
│   ├── github_collector.py     # 1. GitHub API → raw signals
│   ├── preprocessing.py        # 2. clean + derived features
│   ├── summarization.py        # 3. text representation
│   ├── llm_labeling.py         # 4. Groq LLM weak labels
│   ├── utils.py                # 5. stratified split
│   ├── train.py                # 6. DistilBERT fine-tune (Colab) + export preds
│   ├── evaluation.py           # 7. metrics + error tables
│   └── visualization.py        # 8. figures
├── notebooks/
│   └── train_colab.ipynb       # run train.py on a free Colab T4 GPU
├── data/                       # raw / processed / labeled / splits
├── models/trained_models/      # fine-tuned model (gitignored, ~268MB)
├── output/                     # predictions / metrics / tables / figures
└── video/link.txt              # 5-minute pitch video link
```

---

## Setup & reproduction

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GITHUB_TOKEN and GROQ_API_KEY
```

Run the pipeline (steps that need no GPU run locally):

```bash
python src/github_collector.py     # 1. collect (needs GITHUB_TOKEN)
python src/preprocessing.py        # 2. clean
python src/summarization.py        # 3. text
python src/llm_labeling.py         # 4. weak labels (needs GROQ_API_KEY)
python src/utils.py                # 5. split
# 6. training runs on Colab — open notebooks/train_colab.ipynb (free T4 GPU),
#    run all cells, download trained_artifacts.zip, and extract:
#      models/trained_models/distilbert/  and  output/predictions/*.csv
python src/evaluation.py           # 7. metrics + tables  (no torch)
python src/visualization.py        # 8. figures           (no torch)
streamlit run app.py               # 9. dashboard         (no torch)
```

**Cost:** the whole project runs at **zero monetary cost** — GitHub API (free token),
Groq free tier (`llama-3.1-8b-instant`, 500K tokens/day), and Google Colab's free T4
GPU for training.

**Secrets:** `GITHUB_TOKEN` and `GROQ_API_KEY` live only in `.env` (gitignored) —
never committed.

---

## Dashboard (`app.py`)

Four tabs, all reading precomputed artifacts (no torch needed):
1. **Problem & Methodology** — objective, sampling, signals, limitations.
2. **Exploratory Analysis** — label distribution, signals by category, per-category averages.
3. **Model Results** — headline metrics, confusion matrix, per-class F1, most-confused pairs.
4. **Interactive Exploration** — search/filter repos, inspect any repo's weak label,
   model prediction, confidence and text representation.

---

## Ethical note

This project estimates the maturity **a repository reflects**, as a triage aid. It is
trained on **noisy weak labels** from metadata alone and must **never** be used to
judge, rank or make automated decisions about individuals.

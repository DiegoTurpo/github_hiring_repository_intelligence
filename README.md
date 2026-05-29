# github_hiring_repository_intelligence

**Track A — Hiring-Oriented Repository Intelligence**

A weak-supervision NLP pipeline that analyzes GitHub repositories and estimates the
**engineering maturity** they reflect (intern, junior, senior, lead/architect,
template/boilerplate, low-value). The goal is not to judge developers, but to estimate
the engineering complexity and maturity reflected by the repository itself.

## Pipeline

1. **GitHub data collection** — extract repository-level signals via the GitHub API.
2. **Repository representation** — turn metadata into text usable by LLMs and BERT.
3. **LLM weak labeling** — an LLM acts as the initial annotator of categories.
4. **Train / validation / test split** — 70 / 15 / 15, stratified.
5. **BERT fine-tuning** — fine-tune a lightweight transformer (DistilBERT).
6. **Evaluation & error analysis** — accuracy, precision, recall, F1, confusion matrix.

## Project structure

```
github_hiring_repository_intelligence/
├── app.py                   # Streamlit app (4 tabs)
├── README.md
├── requirements.txt
├── src/
│   ├── github_collector.py
│   ├── preprocessing.py
│   ├── summarization.py
│   ├── llm_labeling.py
│   ├── train.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py
├── data/        # raw / processed / labeled / splits
├── models/trained_models/
├── output/      # figures / tables / metrics
└── video/link.txt
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in GITHUB_TOKEN and GEMINI_API_KEY
```

> Full methodology, findings, metrics and business applications are documented at the
> end of the project.

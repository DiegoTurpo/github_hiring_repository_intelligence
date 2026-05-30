"""BERT fine-tuning for repository maturity classification.

Reads  : data/splits/{train,val}.csv
Writes : models/trained_models/distilbert/   (model + tokenizer)

Model  : distilbert-base-uncased  (lightweight, fast to fine-tune)
Input  : the textual repository representation ('text' column)
Output : one of six maturity categories (label_id 0..5)

Because the dataset is imbalanced (lead/template dominate, intern/low_value are
rare), we weight the loss by inverse class frequency so minority categories are
not ignored.

This is meant to run on a GPU (e.g. Google Colab free tier). It also runs on
CPU, just slower.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

MODEL_NAME = "distilbert-base-uncased"
OUT_DIR = "models/trained_models/distilbert"
MAX_LEN = 256
EPOCHS = 5
BATCH = 16
LR = 2e-5

CATEGORIES = ["intern", "junior", "senior", "lead", "template", "low_value"]
ID2LABEL = {i: c for i, c in enumerate(CATEGORIES)}
LABEL2ID = {c: i for i, c in enumerate(CATEGORIES)}


def load_split(path):
    df = pd.read_csv(path)
    return Dataset.from_pandas(df[["text", "label_id"]].rename(
        columns={"label_id": "labels"}))


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro",
                                            zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro",
                                     zero_division=0),
    }


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy to handle imbalance."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(model.device))
        loss = loss_fct(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def main():
    train_df = pd.read_csv("data/splits/train.csv")
    counts = train_df["label_id"].value_counts().sort_index()
    weights = torch.tensor(
        [len(train_df) / (len(CATEGORIES) * counts.get(i, 1))
         for i in range(len(CATEGORIES))], dtype=torch.float)
    print("class weights:", weights.tolist())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

    train_ds = load_split("data/splits/train.csv").map(tok, batched=True)
    val_ds = load_split("data/splits/val.csv").map(tok, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(CATEGORIES),
        id2label=ID2LABEL, label2id=LABEL2ID)

    args = TrainingArguments(
        output_dir="models/checkpoints",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LR,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
        report_to="none",
    )

    trainer = WeightedTrainer(
        class_weights=weights,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()
    print("\nValidation metrics:", trainer.evaluate())

    os.makedirs(OUT_DIR, exist_ok=True)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"\nSaved fine-tuned model -> {OUT_DIR}")

    # Export predictions so evaluation + the Streamlit app need no torch locally.
    export_predictions(trainer, tokenizer, tok)


def export_predictions(trainer, tokenizer, tok):
    """Run inference on the test set and the full dataset; save to output/."""
    os.makedirs("output/predictions", exist_ok=True)

    for name, path, out in [
        ("test", "data/splits/test.csv", "output/predictions/test_predictions.csv"),
        ("all", "data/labeled/labeled.csv", "output/predictions/all_predictions.csv"),
    ]:
        df = pd.read_csv(path)
        if "label_id" not in df.columns:
            df["label_id"] = df["label"].map(LABEL2ID)
        ds = Dataset.from_pandas(df[["text"]]).map(tok, batched=True)
        logits = trainer.predict(ds).predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        pred_ids = probs.argmax(axis=-1)

        df_out = df[["full_name", "text", "label"]].copy()
        df_out["true_label"] = df["label"]
        df_out["pred_label"] = [ID2LABEL[i] for i in pred_ids]
        df_out["pred_confidence"] = probs.max(axis=-1).round(4)
        df_out.to_csv(out, index=False)
        print(f"Saved {name} predictions ({len(df_out)}) -> {out}")


if __name__ == "__main__":
    main()

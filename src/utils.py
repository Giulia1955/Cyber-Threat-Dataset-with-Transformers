import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import AutoTokenizer

from model import DEFAULT_MODEL_NAME

_TOKENIZER_CACHE = {}


def get_tokenizer(model_name=DEFAULT_MODEL_NAME):
    if model_name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE[model_name]


def preprocess_function(examples, max_length=128, model_name=DEFAULT_MODEL_NAME):
    tokenizer = get_tokenizer(model_name=model_name)
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )


def build_preprocess_function(model_name=DEFAULT_MODEL_NAME, max_length=128):
    def _preprocess(examples):
        return preprocess_function(
            examples=examples,
            max_length=max_length,
            model_name=model_name,
        )

    return _preprocess


def compute_multilabel_metrics_from_logits(logits, labels):
    logits_tensor = torch.as_tensor(logits, dtype=torch.float32)
    labels_array = np.asarray(labels)
    probs = torch.sigmoid(logits_tensor).numpy() #funcao de ativacao aqui --------------------------------

    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.3, 0.71, 0.05):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(labels_array, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    predictions = (probs >= best_thresh).astype(int)

    f1 = f1_score(labels_array, predictions, average="macro", zero_division=0)
    precision = precision_score(labels_array, predictions, average="macro", zero_division=0)
    recall = recall_score(labels_array, predictions, average="macro", zero_division=0)

    return {
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall,
        "best_thresh": best_thresh,
    }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return compute_multilabel_metrics_from_logits(logits=logits, labels=labels)

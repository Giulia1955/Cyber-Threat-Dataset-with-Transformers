import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

from utils import compute_multilabel_metrics_from_logits


def _multilabel_loss(logits, labels, pos_weight=None, focal_gamma=0.0):
    bce = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=pos_weight,
        reduction="none",
    )
    if focal_gamma > 0:
        p_t = torch.exp(-bce)
        bce = ((1 - p_t) ** focal_gamma) * bce
    return bce.mean()


def _run_eval(model, dataloader, device, pos_weight=None, focal_gamma=0.0):
    model.eval()
    losses = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = _multilabel_loss(
                logits=logits,
                labels=labels,
                pos_weight=pos_weight,
                focal_gamma=focal_gamma,
            )
            losses.append(loss.item())
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    labels_np = np.concatenate(all_labels, axis=0)
    metrics = compute_multilabel_metrics_from_logits(logits_np, labels_np)
    return float(np.mean(losses)), metrics


def train_lstm_model(
    model,
    train_dataloader,
    eval_dataloader,
    device,
    learning_rate=2e-4,
    weight_decay=0.05,
    num_epochs=30,
    max_grad_norm=1.0,
    focal_gamma=2.0,
    pos_weight=None,
    early_stopping_patience=3,
):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.to(device)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)

    best_f1 = -1.0
    best_state_dict = None
    patience_count = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_losses = []

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = _multilabel_loss(
                logits=logits,
                labels=labels,
                pos_weight=pos_weight,
                focal_gamma=focal_gamma,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            train_losses.append(loss.item())

        eval_loss, eval_metrics = _run_eval(
            model=model,
            dataloader=eval_dataloader,
            device=device,
            pos_weight=pos_weight,
            focal_gamma=focal_gamma,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "eval_loss": eval_loss,
            "eval_f1_macro": eval_metrics["f1_macro"],
            "eval_precision_macro": eval_metrics["precision_macro"],
            "eval_recall_macro": eval_metrics["recall_macro"],
            "eval_best_thresh": eval_metrics["best_thresh"],
            "learning_rate": learning_rate,
        }
        history.append(row)

        if eval_metrics["f1_macro"] > best_f1:
            best_f1 = eval_metrics["f1_macro"]
            best_state_dict = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= early_stopping_patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history

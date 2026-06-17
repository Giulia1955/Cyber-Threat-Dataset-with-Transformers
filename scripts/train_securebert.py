import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from model import get_model
from trainer import CustomTrainer
from utils import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treina o SecureBERT (HF) para classificação multilabel."
    )
    parser.add_argument("--model-name", default="ehsanaghaei/SecureBERT")
    parser.add_argument("--dataset-path", default=str(ROOT / "data/processed/dataset_hf_processado_securebert"))
    parser.add_argument("--label-map-path", default=str(ROOT / "data/processed/label_map.json"))
    parser.add_argument("--pos-weights-path", default=str(ROOT / "data/processed/pos_weights.pt"))
    parser.add_argument("--output-dir", default=str(ROOT / "models/securebert-base-cyber-threat"))
    parser.add_argument("--history-dir", default=str(ROOT / "data/processed/historic"))
    parser.add_argument("--experiment-name", default="securebert_base_lr2e5_ep30_wd05_earlystop3_bs8")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--disable-pos-weights", action="store_true")
    return parser.parse_args()


def main():
    cli_args = parse_args()

    dataset_full = load_from_disk(cli_args.dataset_path)
    datasets_split = dataset_full.train_test_split(test_size=cli_args.test_size, seed=cli_args.seed)
    dataset_train_hf = datasets_split["train"]
    dataset_test_hf = datasets_split["test"]

    with open(cli_args.label_map_path, "r", encoding="utf-8") as f:
        id2label = json.load(f)
    label2id = {v: k for k, v in id2label.items()}
    num_labels = len(id2label)

    model = get_model(
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        model_name=cli_args.model_name,
    )

    output_dir = Path(cli_args.output_dir)
    history_dir = Path(cli_args.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cli_args.learning_rate,
        per_device_train_batch_size=cli_args.train_batch_size,
        per_device_eval_batch_size=cli_args.eval_batch_size,
        num_train_epochs=cli_args.epochs,
        weight_decay=cli_args.weight_decay,
        warmup_ratio=cli_args.warmup_ratio,
        max_grad_norm=cli_args.max_grad_norm,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        seed=cli_args.seed,
    )

    class SaveHistoryCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.log_history:
                pd.DataFrame(state.log_history).to_csv(
                    history_dir / f"{cli_args.experiment_name}.csv",
                    index=False,
                )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train_hf,
        eval_dataset=dataset_test_hf,
        compute_metrics=compute_metrics,
        focal_gamma=cli_args.focal_gamma,
        use_pos_weights=not cli_args.disable_pos_weights,
        pos_weights_path=cli_args.pos_weights_path,
        callbacks=[
            SaveHistoryCallback(),
            EarlyStoppingCallback(early_stopping_patience=cli_args.early_stopping_patience),
        ],
    )

    trainer.train()

    print(f"Treino finalizado para {cli_args.model_name}")
    print(f"Modelo salvo em: {output_dir}")
    print(f"Histórico salvo em: {history_dir / f'{cli_args.experiment_name}.csv'}")


if __name__ == "__main__":
    main()

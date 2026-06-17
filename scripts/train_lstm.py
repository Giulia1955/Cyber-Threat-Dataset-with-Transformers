import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from lstm_data import LSTMDataset, infer_lstm_data_config, save_lstm_data_config
from lstm_model import BiLSTMForMultiLabelClassification
from lstm_trainer import train_lstm_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Treino BiLSTM para classificação multi-label em ameaças cibernéticas."
    )
    parser.add_argument("--dataset-path", default=str(ROOT / "data/processed/dataset_hf_processado"))
    parser.add_argument("--label-map-path", default=str(ROOT / "data/processed/label_map.json"))
    parser.add_argument("--pos-weights-path", default=str(ROOT / "data/processed/pos_weights.pt"))
    parser.add_argument("--model-output-dir", default=str(ROOT / "models/lstm-cyber-threat"))
    parser.add_argument("--history-dir", default=str(ROOT / "data/processed/historic"))
    parser.add_argument("--experiment-name", default="lstm_lr2e4_ep30_wd05_focal2_earlystop3_bs16")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--disable-pos-weights", action="store_true")

    parser.add_argument("--embedding-dim", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unidirectional", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset_full = load_from_disk(args.dataset_path)
    split = dataset_full.train_test_split(test_size=args.test_size, seed=args.seed)
    train_hf = split["train"]
    test_hf = split["test"]

    with open(args.label_map_path, "r", encoding="utf-8") as f:
        id2label = json.load(f)
    num_labels = len(id2label)

    data_cfg = infer_lstm_data_config(dataset_full, pad_token_id=0)
    save_lstm_data_config(data_cfg, ROOT / "data/processed/lstm_data_config.json")

    train_ds = LSTMDataset(train_hf)
    test_ds = LSTMDataset(test_hf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = BiLSTMForMultiLabelClassification(
        vocab_size=data_cfg["vocab_size"],
        num_labels=num_labels,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=not args.unidirectional,
        padding_idx=data_cfg["padding_idx"],
    )

    pos_weight = None
    if not args.disable_pos_weights and os.path.exists(args.pos_weights_path):
        pos_weight = torch.load(args.pos_weights_path, map_location="cpu", weights_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, history = train_lstm_model(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.epochs,
        max_grad_norm=args.max_grad_norm,
        focal_gamma=args.focal_gamma,
        pos_weight=pos_weight,
        early_stopping_patience=args.early_stopping_patience,
    )

    model_output_dir = Path(args.model_output_dir)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_output_dir / "pytorch_model.bin")
    with (model_output_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "bilstm_multilabel",
                "vocab_size": data_cfg["vocab_size"],
                "num_labels": num_labels,
                "embedding_dim": args.embedding_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "bidirectional": not args.unidirectional,
                "padding_idx": data_cfg["padding_idx"],
                "id2label": id2label,
            },
            f,
            ensure_ascii=True,
            indent=2,
        )

    history_dir = Path(args.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(history_dir / f"{args.experiment_name}.csv", index=False)

    print(f"Modelo salvo em: {model_output_dir}")
    print(f"Histórico salvo em: {history_dir / f'{args.experiment_name}.csv'}")


if __name__ == "__main__":
    main()

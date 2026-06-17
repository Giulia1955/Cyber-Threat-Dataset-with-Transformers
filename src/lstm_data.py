import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class LSTMDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        row = self.hf_dataset[idx]
        return {
            "input_ids": torch.tensor(row["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(row["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(row["labels"], dtype=torch.float),
        }


def infer_lstm_data_config(hf_dataset, pad_token_id=0):
    vocab_size = 1
    max_length = 0
    for row in hf_dataset:
        token_ids = row["input_ids"]
        if token_ids:
            vocab_size = max(vocab_size, max(token_ids) + 1)
            max_length = max(max_length, len(token_ids))
    return {
        "vocab_size": vocab_size,
        "max_length": max_length,
        "padding_idx": pad_token_id,
    }


def save_lstm_data_config(config_dict, output_path):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(config_dict, file, ensure_ascii=True, indent=2)

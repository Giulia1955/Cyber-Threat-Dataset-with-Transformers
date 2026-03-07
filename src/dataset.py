# --- src/dataset.py ---
import torch
from transformers import AutoTokenizer

# Opcional: Você pode inicializar o tokenizer aqui ou passá-lo como argumento
tokenizer = AutoTokenizer.from_pretrained("ehsanaghaei/SecureBERT")

def preprocess_function(examples):
    """Função do Passo 2 para tokenizar os textos"""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

class CyberThreatDataset(torch.utils.data.Dataset):
    """Classe do Passo 3 para criar o dataset do PyTorch"""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Importante: Labels devem ser Float para BCEWithLogitsLoss
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)
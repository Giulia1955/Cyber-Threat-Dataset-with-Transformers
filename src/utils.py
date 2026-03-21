from transformers import AutoTokenizer
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

MODEL_NAME = "cisco-ai/SecureBERT2.0-base"
# Inicialize FORA da função para carregar apenas uma vez na memória
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )

# Aproveitando o arquivo utils, vamos criar as métricas para Multi-Label
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # 1. Aplica a sigmoide para transformar logits em probabilidades (0 a 1)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(logits))
    
    # 2. Define o threshold de 0.5 para decidir se a classe está presente ou não
    predictions = (probs >= 0.5).int().numpy()
    
    # 3. Calcula as métricas (usando 'macro' ou 'micro' para multi-label)
    f1 = f1_score(labels, predictions, average='macro', zero_division=0)
    precision = precision_score(labels, predictions, average='macro', zero_division=0)
    recall = recall_score(labels, predictions, average='macro', zero_division=0)
    
    return {
        "f1_macro": f1,
        "precision_macro": precision,
        "recall_macro": recall
    }
















'''from transformers import AutoTokenizer
from sklearn.metrics import f1_score, roc_auc_score, hamming_loss
import numpy as np

model_name = "cisco-ai/SecureBERT2.0-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):

    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))
    predictions = (probs > 0.5).astype(float)
    
    return {
        "f1_micro": f1_score(labels, predictions, average="micro"),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "hamming_loss": hamming_loss(labels, predictions)
    }'''
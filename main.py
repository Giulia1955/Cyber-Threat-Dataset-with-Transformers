import pandas as pd

#print(ds.columns) 
#print(ds.head())
#print(ds.shape)
#print(ds.value_counts('label_3'))

from sklearn.feature_extraction.text import CountVectorizer

'''
#criação da BOW
vetorize = CountVectorizer(lowercase = False, max_features = 50)
bag_of_words = vetorize.fit_transform(ds["text"])
#print(bag_of_words.shape)
sparse_matrix_text = pd.DataFrame.sparse.from_spmatrix(bag_of_words, columns = vetorize.get_feature_names_out())
#print (sparse_matrix_text)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bag_of_words, ds.text)

from sklearn.linear_model import LogisticRegression
linear_regression = LogisticRegression()
linear_regression.fit(x_train, y_train)
accuracy = linear_regression.score(x_test, y_test)
print(accuracy) '''

# Importações essenciais
import re
import shutil
import unicodedata
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import Counter
import polars as pl
from tqdm import tqdm
import json
import random
import warnings 
from datasets import load_dataset

from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts 
from torch.amp import autocast, GradScaler 

device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

#from huggingface_hub import login
#login(token = 'insira token aqui')

def __init__(self, dataset_path: str = "Dataset/Cyber-Threat-Intelligence-Custom-Data_new_processed.csv"):
    cols_to_keep = [
        "id", "text", "relations", "diagnosis",
        "solutions", "label_1", "label_2", "label_3"
    ]
    self.df = pl.read_csv(dataset_path, columns=cols_to_keep)
    self.texts = self.df["text"]
    print(f"Carregado: {len(self.df):,} textos")


def clean_cti_text(text: str) -> str:
    if not text: return ""
    text = text.lower()
    
    # Mascaramento de IoCs (Essencial para manter o contexto técnico)
    text = re.sub(r'\b\d{1,3}(?:\.\d{1,3}){3}\b', ' [IP] ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' [URL] ', text)
    text = re.sub(r'\b[a-fA-F0-9]{32,64}\b', ' [HASH] ', text)
    
    # Limpeza de ruído
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def prepare_t5_input(row: dict) -> str:
    """Input para o T5 com prefixo de tarefa"""
    return f"analyze threat: {row['text_cleaned']}"

def prepare_t5_target(row: dict) -> str:
    """
    Cria um alvo estruturado para o T5.
    Exemplo: 'L1: Malware, L2: Ransomware, L3: WannaCry'
    """
    return f"L1: {row['label_1']}, L2: {row['label_2']}, L3: {row['label_3']}"

# 1. Carregando o dataset (ajuste o caminho se necessário)
df = pl.read_csv("Dataset/Cyber-Threat-Intelligence-Custom-Data_new_processed.csv")

# 2. Processamento Principal
df_processed = (
    df
    .select(["text", "label_1", "label_2", "label_3"])
    .drop_nulls()
    .with_columns([
        # Limpeza comum para todos os modelos
        pl.col("text").map_elements(clean_cti_text, return_dtype=pl.String).alias("text_cleaned")
    ])
    .with_columns([
        # Preparação específica para T5 (Text-to-Text)
        pl.struct(["text_cleaned"]).map_elements(prepare_t5_input, return_dtype=pl.String).alias("t5_input"),
        pl.struct(["label_1", "label_2", "label_3"]).map_elements(prepare_t5_target, return_dtype=pl.String).alias("t5_target")
    ])
)

# 3. Visualização do resultado
print("Exemplo de dado processado para os modelos:")
print(df_processed.select(["t5_input", "t5_target"]).head(2))


#tokenização dos dados
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

# Ajustando e transformando as labels
# Convertemos cada coluna de categorias em uma sequência de números (0, 1, 2...)
y1 = le1.fit_transform(df_processed["label_1"])
y2 = le2.fit_transform(df_processed["label_2"])
y3 = le3.fit_transform(df_processed["label_3"])

# Guardando o número de classes
num_classes = {
    "L1": len(le1.classes_),
    "L2": len(le2.classes_),
    "L3": len(le3.classes_)
}

print(f"Classes identificadas: {num_classes}")

# Exemplo de como isso vira um tensor para o PyTorch
labels_tensor = torch.tensor(list(zip(y1, y2, y3)), dtype=torch.long)
# labels_tensor agora tem o formato [n_exemplos, 3]
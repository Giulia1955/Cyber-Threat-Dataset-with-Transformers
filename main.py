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

from transformers import AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts 
from torch.amp import autocast, GradScaler 

device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Usando: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

from huggingface_hub import login
login(token = 'insira token aqui')

def __init__(self, dataset_path: str = "Dataset/Cyber-Threat-Intelligence-Custom-Data_new_processed.csv"):
    cols_to_keep = [
        "id", "text", "relations", "diagnosis",
        "solutions", "label_1", "label_2", "label_3"
    ]
    self.df = pl.read_csv(dataset_path, columns=cols_to_keep)
    self.texts = self.df["text"]
    print(f"Carregado: {len(self.df):,} textos")


def load_all(
    self,
    char_limit: int = 500_000,
    sep_token: str = "[SEP]",
    insert_sep_final: bool = True
) -> str:

    df_clean = (
        self.df
        .filter(pl.col("text").str.len_chars() > 50)
        .with_columns([
            pl.col("text")
            # 1. Templates Wiki
            .str.replace_all(r'\{\{[^}]*\}\}', ' ')

            # 2. URLs e Emails
            .str.replace_all(r'https?://\S+|www\.\S+', '[URL]')
            .str.replace_all(r'\S+@\S+', '[EMAIL]')

            # 3. Normalização de pontuação
            .str.replace_all(r'[“”„‟]', '"')
            .str.replace_all(r"[‘’‚‛]", "'")

            # 4. Limpeza de espaços
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
        ])
        .filter(
            # Critérios de CTI: textos de cibersegurança podem ser densos em símbolos
            (pl.col("text").str.len_chars() > 30) &
            (pl.col("text").str.len_chars() < 10_000)
        )
    )

    texts = df_clean.get_column("text").to_list()

    return texts

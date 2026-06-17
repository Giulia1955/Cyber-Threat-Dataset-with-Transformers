import torch
from torch import nn


class BiLSTMForMultiLabelClassification(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        embedding_dim=200,
        hidden_size=256,
        num_layers=2,
        dropout=0.3,
        bidirectional=True,
        padding_idx=0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout,
        )

        self.dropout = nn.Dropout(dropout)
        output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.classifier = nn.Linear(output_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        embeddings = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=input_ids.shape[1],
        )

        mask = attention_mask.unsqueeze(-1).float()
        masked_output = lstm_output * mask
        pooled = masked_output.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-8)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits

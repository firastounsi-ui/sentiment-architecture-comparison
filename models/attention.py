import torch
import torch.nn as nn


class AttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx=0):
        super().__init__()

        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.attention = nn.Linear(embed_dim, 1)

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)

        attention_scores = self.attention(embedded).squeeze(-1)

        pad_mask = input_ids == self.pad_idx

        attention_scores = attention_scores.masked_fill(pad_mask, -1e9)

        attention_weights = torch.softmax(attention_scores, dim=1)

        weighted_embeddings = embedded * attention_weights.unsqueeze(-1)

        pooled = weighted_embeddings.sum(dim=1)

        logits = self.classifier(pooled)

        return logits

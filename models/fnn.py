import torch
import torch.nn as nn


class FNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)

        pooled = embedded.mean(dim=1)

        logits = self.classifier(pooled)

        return logits

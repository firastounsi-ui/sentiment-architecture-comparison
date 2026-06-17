import torch
from torch.utils.data import Dataset


class IMDBTorchDataset(Dataset):
    def __init__(self, examples, vocab, max_length):
        self.examples = examples
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text, label = self.examples[idx]

        input_ids = self.vocab.encode(text, self.max_length)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
        }

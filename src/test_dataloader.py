from torch.utils.data import DataLoader

from src.dataset import load_imdb_dataset
from src.vocab import Vocabulary
from src.torch_dataset import IMDBTorchDataset
from models.fnn import FNNClassifier

train_examples, test_examples = load_imdb_dataset("data/aclImdb")

train_texts = [text for text, label in train_examples]

vocab = Vocabulary(max_size=20000, min_freq=2)
vocab.build(train_texts)

train_dataset = IMDBTorchDataset(
    examples=train_examples,
    vocab=vocab,
    max_length=200
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

batch = next(iter(train_loader))

print("Input IDs batch shape:", batch["input_ids"].shape)
print("Labels batch shape:", batch["label"].shape)

model = FNNClassifier(
    vocab_size=len(vocab),
    embed_dim=128,
    num_classes=2
)

outputs = model(batch["input_ids"])

print("Model output shape:", outputs.shape)

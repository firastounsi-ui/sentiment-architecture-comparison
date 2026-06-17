from dataset import load_imdb_dataset
from vocab import Vocabulary
from torch_dataset import IMDBTorchDataset


train_examples, test_examples = load_imdb_dataset("data/aclImdb")

train_texts = [text for text, label in train_examples]

vocab = Vocabulary(max_size=20000, min_freq=2)
vocab.build(train_texts)

torch_train = IMDBTorchDataset(
    examples=train_examples,
    vocab=vocab,
    max_length=200
)

sample = torch_train[0]

print("Vocab size:", len(vocab))
print("Dataset size:", len(torch_train))
print("Input IDs shape:", sample["input_ids"].shape)
print("Label:", sample["label"])
print("First 20 IDs:", sample["input_ids"][:20])

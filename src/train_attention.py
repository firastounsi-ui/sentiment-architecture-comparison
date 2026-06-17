import random
import torch
from torch.utils.data import DataLoader

from src.dataset import load_imdb_dataset
from src.vocab import Vocabulary
from src.torch_dataset import IMDBTorchDataset
from models.attention import AttentionClassifier
from src.evaluate_metrics import evaluate_metrics

def calculate_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct, total


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct, total = calculate_accuracy(logits, labels)

        total_loss += loss.item() * total
        total_correct += correct
        total_examples += total

    return total_loss / total_examples, total_correct / total_examples


def evaluate(model, dataloader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids)
            loss = loss_fn(logits, labels)

            correct, total = calculate_accuracy(logits, labels)

            total_loss += loss.item() * total
            total_correct += correct
            total_examples += total

    return total_loss / total_examples, total_correct / total_examples


def main():
    random.seed(42)
    torch.manual_seed(42)

    data_dir = "data/aclImdb"
    max_length = 200
    vocab_size = 20000
    min_freq = 2
    embed_dim = 100
    num_classes = 2
    batch_size = 32
    num_epochs = 3
    learning_rate = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    print("Loading dataset...")
    train_examples, test_examples = load_imdb_dataset(data_dir)

    random.shuffle(train_examples)

    train_subset = train_examples[:20000]
    val_subset = train_examples[20000:]

    print("Train subset size:", len(train_subset))
    print("Validation subset size:", len(val_subset))
    print("Official test set size:", len(test_examples))

    print("Building vocabulary from train subset only...")
    train_texts = [text for text, label in train_subset]

    vocab = Vocabulary(max_size=vocab_size, min_freq=min_freq)
    vocab.build(train_texts)

    print("Vocabulary size:", len(vocab))

    train_dataset = IMDBTorchDataset(train_subset, vocab, max_length)
    val_dataset = IMDBTorchDataset(val_subset, vocab, max_length)
    test_dataset = IMDBTorchDataset(test_examples, vocab, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AttentionClassifier(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        num_classes=num_classes,
        pad_idx=0
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("\nStarting attention training...")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device
        )

        val_loss, val_accuracy = evaluate(
            model, val_loader, loss_fn, device
        )

        print(
            f"Epoch {epoch}: "
            f"Train Loss = {train_loss:.4f}, "
            f"Train Accuracy = {train_accuracy:.4f}, "
            f"Val Loss = {val_loss:.4f}, "
            f"Val Accuracy = {val_accuracy:.4f}"
        )

    test_loss, test_accuracy = evaluate(
        model, test_loader, loss_fn, device
    )
    metrics = evaluate_metrics(model, test_loader, device)

    print("\nFinal Test Metrics")
    print(f"Accuracy  = {metrics['accuracy']:.4f}")
    print(f"Precision = {metrics['precision']:.4f}")
    print(f"Recall    = {metrics['recall']:.4f}")
    print(f"F1 Score  = {metrics['f1']:.4f}")
    torch.save(model.state_dict(), "results/models/attention.pt")
    print("Saved model to results/models/attention.pt")

if __name__ == "__main__":
    main()

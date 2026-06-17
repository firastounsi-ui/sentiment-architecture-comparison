from pathlib import Path
import random


def load_reviews_from_folder(folder_path, label):
    folder = Path(folder_path)
    examples = []

    for file_path in folder.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        examples.append((text, label))

    return examples


def load_imdb_dataset(data_dir):
    data_path = Path(data_dir)

    train_pos = load_reviews_from_folder(data_path / "train" / "pos", 1)
    train_neg = load_reviews_from_folder(data_path / "train" / "neg", 0)

    test_pos = load_reviews_from_folder(data_path / "test" / "pos", 1)
    test_neg = load_reviews_from_folder(data_path / "test" / "neg", 0)

    train_examples = train_pos + train_neg
    test_examples = test_pos + test_neg

    random.shuffle(train_examples)
    random.shuffle(test_examples)

    return train_examples, test_examples


if __name__ == "__main__":
    train, test = load_imdb_dataset("data/aclImdb")

    print("Train examples:", len(train))
    print("Test examples:", len(test))

    print("\nFirst training example:")
    print("Label:", train[0][1])
    print("Text preview:", train[0][0][:300])

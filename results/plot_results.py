import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_FILE = "results/metrics.csv"
OUTPUT_DIR = Path("results/figures")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_FILE)


def save_bar(column, title, ylabel, filename):
    plt.figure(figsize=(8, 5))
    plt.bar(df["model"], df[column])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename)
    plt.close()


save_bar(
    "test_accuracy",
    "Test Accuracy by Model",
    "Accuracy",
    "test_accuracy.png"
)

save_bar(
    "f1",
    "F1 Score by Model",
    "F1 Score",
    "f1_score.png"
)

save_bar(
    "precision",
    "Precision by Model",
    "Precision",
    "precision.png"
)

save_bar(
    "recall",
    "Recall by Model",
    "Recall",
    "recall.png"
)

x = range(len(df))
width = 0.2

plt.figure(figsize=(10, 5))

plt.bar(
    [i - 1.5 * width for i in x],
    df["test_accuracy"],
    width=width,
    label="Accuracy"
)

plt.bar(
    [i - 0.5 * width for i in x],
    df["precision"],
    width=width,
    label="Precision"
)

plt.bar(
    [i + 0.5 * width for i in x],
    df["recall"],
    width=width,
    label="Recall"
)

plt.bar(
    [i + 1.5 * width for i in x],
    df["f1"],
    width=width,
    label="F1"
)

plt.xticks(x, df["model"])
plt.title("Test Metrics by Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "test_metrics_comparison.png")
plt.close()

x = range(len(df))
width = 0.25

plt.figure(figsize=(10, 5))

plt.bar(
    [i - width for i in x],
    df["train_accuracy"],
    width=width,
    label="Train"
)

plt.bar(
    x,
    df["val_accuracy"],
    width=width,
    label="Validation"
)

plt.bar(
    [i + width for i in x],
    df["test_accuracy"],
    width=width,
    label="Test"
)

plt.xticks(x, df["model"])
plt.title("Train, Validation, and Test Accuracy")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "accuracy_comparison.png")
plt.close()

print("Figures saved to:")
print(OUTPUT_DIR.resolve())

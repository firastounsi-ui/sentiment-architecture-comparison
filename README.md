# Sentiment Architecture Comparison

This project compares different neural network architectures for sentiment analysis on the IMDB movie review dataset using PyTorch.

## Models

The following architectures were implemented and evaluated:

- Feedforward Neural Network (FNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory Network (LSTM)
- Attention-Based Neural Network

All models use the same preprocessing pipeline, vocabulary construction, training split, and optimization settings to ensure a fair comparison.

## Dataset

The project uses the IMDB movie review dataset.

Dataset split:

| Split | Reviews |
|---------|---------:|
| Training | 20,000 |
| Validation | 5,000 |
| Test | 25,000 |

## Project Structure

```text
models/
├── fnn.py
├── rnn.py
├── lstm.py
└── attention.py

src/
├── dataset.py
├── vocab.py
├── torch_dataset.py
├── evaluate_metrics.py
├── train_fnn.py
├── train_rnn.py
├── train_lstm.py
└── train_attention.py

results/
├── metrics.csv
├── figures/
└── models/

report/
└── report.pdf
```

## Training Models

Train individual models:

```bash
python3 -m src.train_fnn
python3 -m src.train_rnn
python3 -m src.train_lstm
python3 -m src.train_attention
```

Run all experiments:

```bash
./run_all.sh
```

## Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

## Results

| Model | Test Accuracy |
|---------|---------:|
| FNN | 83.89% |
| RNN | 49.89% |
| LSTM | 68.25% |
| Attention | 83.58% |

The Attention model achieved the highest F1-score (84.15%), while the FNN achieved the highest test accuracy (83.89%).

## Generated Figures

Performance plots can be generated with:

```bash
python3 results/plot_results.py
```

Generated figures are stored in:

```text
results/figures/
```

## Report

The full analysis and discussion of the results can be found in:

```text
report/report.pdf
```

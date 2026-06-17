#!/bin/bash
set -e

python3 -m src.train_fnn
python3 -m src.train_rnn
python3 -m src.train_lstm
python3 -m src.train_attention

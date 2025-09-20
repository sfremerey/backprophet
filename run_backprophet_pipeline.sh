#!/usr/bin/env bash
cd /home/sfremerey/backprophet
mkdir -p logs
git pull --ff-only
./.venv/bin/python 1_datacrawler.py
export MKL_DEBUG_CPU_TYPE=5
./.venv/bin/python 2_simple_mlp.py
./.venv/bin/python 3_lstm.py
./.venv/bin/python 4_gru.py
./.venv/bin/python 5_rnn.py
./.venv/bin/python 6_cnn.py
./.venv/bin/python 7_ensemble.py
./.venv/bin/python 8_sentiment.py
git add data/META_predictions.csv data/META_sentiment.csv || true
git commit -m "Add recent predictions and sentiment analysis." || true
git push origin master || true

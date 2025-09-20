#/bin/bash
cd /home/sfremerey/backprophet
git pull
source .venv/bin/activate
python 1_datacrawler.py
python 2_simple_mlp.py
python 3_lstm.py
python 4_gru.py
python 5_rnn.py
python 6_cnn.py
python 7_ensemble.py
python 8_sentiment.py
git add data/META_predictions.csv
git add data/META_sentiment.csv
git commit -m "Add recent predictions and sentiment analysis."
git push origin master

# backprophet
This is backprophet - a deep learning-based tool for stocks prediction.
To assure that the most recent data is crawled, each Python script should be run one after each other in the correct order.
The order is defined by the preceding number of each script.

## Python requirements
This software is tested under Windows 11, Python 3.13.5 and PyTorch using CPU for computations.
At first, it is recommended to install the required Python packages via `pip install`.
You can also call `pip install -r requirements.txt`

## Structure
The project consists of the following files and should be run one after the other.
Then you can also assert that the most recent data is used, ideally from yesterday's close.

* `1_datacrawler.py`: Crawls data from various sources and saves it, e.g.:
	* Yahoo Finance using [yfinance](https://github.com/ranaroussi/yfinance)
	* GPR daily using index data from [Matteo Iacoviello](https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls)
	* Fear & Greed Index using data from CNN API, also cf. [this website](https://edition.cnn.com/markets/fear-and-greed)
* `2_simple_mlp.py`: Trains and evaluates a simple MLP regression model
* `3_lstm.py`: Trains and evaluates a LSTM model
* `4_gru.py`: Trains and evaluates a GRU model
* `5_rnn.py`: Trains and evaluates a RNN model
* `6_cnn.py`: Trains and evaluates a CNN model
* `7_ensemble.py`: Evaluates using an ensemble model, hence a combination of the most promising pre-trained models
* (optional) `8_sentiment.py`: Sends API requests to LLMs to get sentiment data, cf. below for instructions how to run

## Sentiment
If you also want to do a sentimental analysis using LLMs, hence getting a score from 1-100 based on current news (currently only META shares is implemented), you at first need to create a file named `secrets.json` in the root of this project. This is read by the Python script `8_sentiment.py`.
To create the API key for Google Gemini, cf. [here](https://aistudio.google.com/app/apikey) and for Perplexity cf. [here](https://www.perplexity.ai/account/api)
It strictly needs to follow this format:

```json
{
	"PERPLEXITY_API_KEY": "pplx-noiehj9i5wjhe0iqh05",
	"GEMINI_API_KEY": "h98b9ju4wh45u5ejhorjg"
}
```

Especially the sentiment analysis part is experimental.

## License
The contents of this repository follow the [Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0) license.

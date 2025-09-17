# backprophet
This is backprophet - a tool for stocks prediction.
To assure that the most recent data is crawled, each Python script should be run one after each other in the correct order.
The order is defined by the preceding number of each script.

## Python requirements
This software is tested under Windows 11 and Python 3.13.5
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
* ...
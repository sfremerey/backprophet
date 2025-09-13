# backprophet
This is backprophet - a tool for stocks prediction.
To assure that the most recent data is crawled, each Python script should be run one after each other in the correct order.

## Structure
The project consists of the following files and should be run one after the other:

* `1_datacrawler.py`: Crawls stocks data from Yahoo Finance using [yfinance](https://github.com/ranaroussi/yfinance) for given asset and saves it.
* 
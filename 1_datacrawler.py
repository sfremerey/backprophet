import pandas as pd
import yfinance as yf
import requests, urllib

from sklearn.preprocessing import RobustScaler, MinMaxScaler
from fake_useragent import UserAgent


YEARS_BACK = 5  # Number of years to crawl, maximum about 5 as CNN API (Fear & Greed Index) does not go back further
# List of symbols to crawl from yfinance in addition to S&P 500
# For more details, cf. https://finance.yahoo.com/quote/[symbol]
YFINANCE_SYMBOLS = ["^DJI", "META"]
ROLLINGAVG = False
SCALE = False

def get_append_close_volume_from_yfinance(df, symbol, start_date, end_date):
    print(f"Crawl data from yfinance for {symbol}...")
    data = yf.download(symbol, start=start_date, end=end_date, group_by="column", auto_adjust=False, interval="1d", threads=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    # Keep Close & Volume
    df_new = data.loc[:, ["Close", "Volume"]].copy()
    df_new = df_new.reset_index()
    df_new = df_new.rename(columns={
        "Close": f"{symbol}_CLOSE",
        "Volume": f"{symbol}_VOLUME",
        "Date": "DATE"
    })
    df_new["DATE"] = pd.to_datetime(df_new["DATE"]).dt.date
    df_new = df_new.reset_index(drop=True)
    df_new.index.name = None
    df_new.columns.name = None
    if symbol in ["EURUSD=X", "^TNX", "DX-Y.NYB"]:  # In case of some symbols, don't merge the VOLUME as it is empty
        df_return = df.merge(df_new[["DATE", f"{symbol}_CLOSE"]], on="DATE", how="left")
    else:
        df_return = df.merge(df_new[["DATE", f"{symbol}_CLOSE", f"{symbol}_VOLUME"]], on="DATE", how="left")
    return df_return

def get_append_gprd(df):
    urllib.request.urlretrieve("https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls",
                           "data/data_gpr_daily_recent.xls")
    df_gprd = pd.read_excel("data/data_gpr_daily_recent.xls")  # Needs: pip install xlrd
    df_gprd = df_gprd.rename(columns={"date": "DATE"})
    df_gprd["DATE"] = pd.to_datetime(df_gprd["DATE"], format="mixed").dt.date
    df_return = df.merge(df_gprd[["DATE", "GPRD"]], on="DATE", how="left")
    return df_return

def get_append_fearandgreed(df, start_date):
    # Crawls Fear & Greed Index from CNN API
    # Also cf. https://edition.cnn.com/markets/fear-and-greed
    # Partially inspired by https://medium.com/@polish.greg/fear-and-greed-index-python-scraper-96e71e57dbd0
    fearandgreed_base_url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata/"
    ua = UserAgent()
    headers = {
        "User-Agent": ua.random  # Use random user-agent to prevent API errors
    }
    r = requests.get(f"{fearandgreed_base_url}{str(start_date)}", headers=headers)
    if r.status_code == 200:
        data = r.json()["fear_and_greed_historical"]["data"]
        df_fearandgreed = pd.DataFrame(data)
        df_fearandgreed = df_fearandgreed.rename(columns={
            "x": "DATE",
            "y": "FEARANDGREED"
        })
        df_fearandgreed["DATE"] = pd.to_datetime(df_fearandgreed["DATE"].astype(int), unit="ms", utc=True).dt.date
        df_fearandgreed["FEARANDGREED"] = pd.to_numeric(df_fearandgreed["FEARANDGREED"]).astype(float)
        df_return = df.merge(df_fearandgreed[["DATE", "FEARANDGREED"]], on="DATE", how="left")
        return df_return
    elif r.status_code == 500:
        print("Server error 500: internal error at the API. Please choose a smaller YEARS_BACK parameter to add Fear & Greed Index data...")
        return df
    else:
        print(f"Unexpected status {r.status_code} from the CNN API...")
        return df

def main():
    start_date = pd.Timestamp.today() - pd.DateOffset(years=YEARS_BACK)
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.Timestamp.today()
    end_date = pd.to_datetime(end_date).date()
    print(f"Start crawling data from {start_date} to {end_date}...")

    # Start with S&P 500 index to build initial df
    # Reason: get latest official trading day
    print(f"Crawl data from yfinance for ^SPX...")
    data = yf.download("^SPX", start=start_date, end=end_date, group_by="column", auto_adjust=False, interval="1d", threads=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    # Only keep Close & Volume as these are relevant columns
    df = data.loc[:, ["Close", "Volume"]].copy()
    df = df.reset_index()
    df = df.rename(columns={
        "Close": f"^SPX_CLOSE",
        "Volume": f"^SPX_VOLUME",
        "Date": "DATE"
    })
    df["DATE"] = pd.to_datetime(df["DATE"]).dt.date
    df = df.reset_index(drop=True)
    df.index.name = None
    df.columns.name = None

    # Append further stocks data to df
    for symbol in YFINANCE_SYMBOLS:
        df = get_append_close_volume_from_yfinance(df, symbol, start_date, end_date)

    # Append data from geopolitical risk (GPR) index from https://www.matteoiacoviello.com/gpr.htm
    # The daily data (GPRD) are updated every Monday.
    # If the first day of the month or week falls on a federal holiday, data updates will take place the next business day.
    print(f"Crawl data from matteoiacoviello.com for GPRD...")
    df = get_append_gprd(df)

    # Append Fear&Greed data to df
    print(f"Crawl data from CNN API for Fear & Greed Index...")
    df = get_append_fearandgreed(df, start_date)

    # Imputing, dropping NaN values, save df
    print("\n df before imputing and dropping NaN values:")
    df.info()
    df = df.drop_duplicates(subset=["DATE"], keep="first")  # Sometimes we get duplicated rows, remove them
    df = df.ffill()  # Impute NA/NaN values by propagating the last valid observation to next valid.
    df = df.dropna()  # Drop all rows with NA/NaN values, could e.g. happen when some of the FRED values are not there for the very 1st row
    print("\n df after imputing and dropping NaN values:")
    df.info()

    # Add 5, 21, 50 and 200 day moving average to df (per column)
    # Choose only numeric columns and exclude the DATE column
    num_cols = [c for c in df.columns
                if c != "DATE" and pd.api.types.is_numeric_dtype(df[c])]
    if ROLLINGAVG:
        for col in num_cols:
            df[f"{col}_rollingavg5"] = df[col].rolling(window=5).mean()
            df[f"{col}_rollingavg21"] = df[col].rolling(window=21).mean()
            df[f"{col}_rollingavg50"] = df[col].rolling(window=50).mean()
            df[f"{col}_rollingavg200"] = df[col].rolling(window=200).mean()

    df = df.dropna()  # Again drop all rows with NA/NaN values, happens due to moving average
    df.sort_values(by="DATE", ascending=True, inplace=True)

    # Save original and scaled versions of data
    print(f"\n Save original data to data/{end_date}.csv ...")
    df.to_csv(f"data/{end_date}.csv",index=False)

    if SCALE:
        # Scale data
        num_cols = [c for c in df.columns
                    if c != "DATE" and pd.api.types.is_numeric_dtype(df[c])]
        # RobustScaler
        robustscaled_values = RobustScaler().fit_transform(df[num_cols])
        df_robustscaled = df.copy()
        df_robustscaled[num_cols] = robustscaled_values
        df_robustscaled.to_csv(f"data/{end_date}_robustscaled.csv", index=False)
        # MinMaxScaler
        minmaxscaled_values = MinMaxScaler().fit_transform(df[num_cols])
        df_minmaxscaled = df.copy()
        df_minmaxscaled[num_cols] = minmaxscaled_values
        df_minmaxscaled.to_csv(f"data/{end_date}_minmaxscaled.csv", index=False)


if __name__ == "__main__":
    main()
    print("\n Done - exiting now...")
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import yfinance as yf
from ta import add_all_ta_features
from ta.trend import SMAIndicator, EMAIndicator
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm
from datetime import timedelta

from src.tradingbot.exceptions import MissingHistoricalDataException, NotEnoughDataException, \
    SentimentDataDownloadFailedException
from src.tradingbot.preprocessor.config import Config


def download_stock_market_data_for_tickers(tickers: List[str], start: str, end: str, attributes=None) -> pd.DataFrame:
    """"
    :param tickers: List of ticker names to download
    :param attributes: List of ticker attributes to keep.
    :param start: Start date in format YYYY-mm-dd
    :param end: End date in format YYYY-mm-dd
    :return: Downloaded dataframe with desired tickers and attributes
    """

    # We will use the Adjusted Close price to take into account all corporate actions,
    # such as stock splits and dividends, to give a more accurate reflection of the true value of the stock
    # and present a coherent picture of returns.
    if attributes is None:
        attributes = ["Adj Close"]

    df_list = []
    failed_download_list: List[str] = []
    for index, ticker in enumerate(tickers):
        ticker_df = yf.download(ticker, start=start, end=end, progress=True)

        if ticker_df.empty:
            # Download not successful - most probably because ticker is named differently on yahoo finance
            failed_download_list.append(ticker)
            continue

        ticker_df = ticker_df.rename(
            columns={"Open": ticker + "_Open", "High": ticker + "_High", "Low": ticker + "_Low",
                     "Close": ticker + "_Close", "Adj Close": ticker + "_Adj Close",
                     "Volume": ticker + "_Volume"})
        df_list.append(ticker_df)
        print(f'Downloaded ticker {index} of {len(tickers)}!')

    ticker_df = pd.concat(df_list, axis=1)
    downloaded_tickers = list(set(tickers) - set(failed_download_list))

    # Keep only desired columns
    if attributes == 'all':
        df_filtered = ticker_df
    else:
        df_filtered = pd.DataFrame()
        for ticker in downloaded_tickers:
            for attr in attributes:
                df_filtered[f"{ticker}_{attr}"] = ticker_df[f"{ticker}_{attr}"]

    # Create entries even for dates when ticker market was closed (all days within particular year)
    all_days = pd.date_range(df_filtered.index.min(), df_filtered.index.max(), freq='D')
    df_filtered = df_filtered.reindex(all_days)

    # Fill in 'holes' in dataframe presenting dates when ticker market was closed
    df_filtered = df_filtered.fillna(method='ffill')

    print('Download finished!')
    print(f"{len(failed_download_list)} Tickers failed to download: {failed_download_list}")

    return df_filtered


class Ticker:
    name: str
    financial_data: Dict
    data_df: pd.DataFrame

    def __init__(self, name: str):
        load_dotenv()
        self.name = name
        self.financial_data = {}

    def fetch_historical_data(self, start='2020-01-01', end='2023-01-01') -> None:
        """
            Fetches the historical data for the ticker
        """
        self.data_df = download_stock_market_data_for_tickers(tickers=[self.name], attributes='all', start=start,
                                                              end=end)
        self.data_df.index.name = 'Date'

    def fetch_financial_data(self) -> None:
        """
            Fetches the historical data for the ticker
        """
        metrics = ["INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "EARNINGS"]
        for metric in metrics:
            url = f"https://www.alphavantage.co/query?function={metric}&symbol={self.name}&apikey={os.getenv('ALPHAVANTAGE_API_KEY')}"
            r = requests.get(url)
            data = r.json()
            self.financial_data[metric] = data

    def compute_technical_indicators(self) -> None:
        """
            Computes the technical indicators for previously fetched historical data
        """
        if not hasattr(self, 'data_df'):
            raise MissingHistoricalDataException

        if len(self.data_df) < 2:
            raise NotEnoughDataException

        self.data_df = add_all_ta_features(
            self.data_df, open=f"{self.name}_Open", high=f"{self.name}_High", low=f"{self.name}_Low",
            close=f"{self.name}_Close", volume=f"{self.name}_Volume")

        self.data_df["volume_vpt_sma"] = SMAIndicator(close=self.data_df["volume_vpt"], window=12).sma_indicator()
        self.data_df["trend_nvi_ema_255"] = EMAIndicator(close=self.data_df["volume_nvi"], window=255).ema_indicator()
        self.data_df["trend_ema_20"] = EMAIndicator(close=self.data_df[f"{self.name}_Close"], window=20).ema_indicator()
        self.data_df["momentum_tsi_ema_12"] = EMAIndicator(close=self.data_df["momentum_tsi"],
                                                           window=12).ema_indicator()
        self.data_df["trend_ema_50"] = EMAIndicator(close=self.data_df[f"{self.name}_Close"], window=50).ema_indicator()
        self.data_df["trend_sma_50"] = SMAIndicator(close=self.data_df[f"{self.name}_Close"], window=50).sma_indicator()
        self.data_df["trend_sma_200"] = SMAIndicator(close=self.data_df[f"{self.name}_Close"],
                                                     window=200).sma_indicator()
        self.data_df["volume_adi_7d_ago"] = self.data_df["volume_adi"].shift(7)
        self.data_df[f"{self.name}_Close_7d_ago"] = self.data_df[f"{self.name}_Adj Close"].shift(7)

    def fetch_sentiment_data(self) -> None:
        """
        Fetch sentiment for a given ticker based on Twitter and Reddit posts
        """

        if not hasattr(self, 'data_df'):
            raise MissingHistoricalDataException

        all_dates = self.data_df.index.tolist()
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=1)
        s.mount('https://', HTTPAdapter(max_retries=retries))
        for date in tqdm(all_dates, desc="Downloading sentiment data..."):
            current_date = date.strftime('%Y-%m-%d')
            following_date = (date + timedelta(days=1)).strftime('%Y-%m-%d')
            url = f"https://api.stockgeist.ai/stock/us/hist/message-metrics?symbols={self.name}&start={current_date}&end={following_date}&timeframe=1d&metrics=total_count%2Cpos_total_count"
            resp = s.get(url, headers={'token': Config.get_value('STOCKGEIST_API_KEY')}).json()
            try:
                sentiment = resp['data'][self.name][0]['pos_total_count'] / resp['data'][self.name][0]['total_count'] if \
                    resp['data'][self.name] else 0.5
            except (ZeroDivisionError, KeyError):
                raise SentimentDataDownloadFailedException(
                    message=f"Error downloading sentiment data for date {date}. Resp: {resp}")
                # sentiment = np.nan
            self.data_df.at[date, 'sentiment'] = sentiment

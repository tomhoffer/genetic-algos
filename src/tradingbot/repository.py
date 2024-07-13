import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.tradingbot.config import Config


@dataclass
class TradingdataRepository:

    @lru_cache(maxsize=0 if "pytest" in sys.modules else 100)
    def load_ticker_data(
            self,
            path=Path(__file__).parent / f"./data/data-{Config.get_value('TRADED_TICKER_NAME')}.csv",
            start_date: str = None,
            end_date: str = None,
    ) -> pd.DataFrame:
        if start_date and end_date:
            return pd.read_csv(path, parse_dates=['Date'], index_col=['Date'])[start_date:end_date]
        else:
            return pd.read_csv(path, parse_dates=['Date'], index_col=['Date'])


    def get_latest_record() -> np.array:
        yield

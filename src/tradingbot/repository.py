import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from src.tradingbot.config import Config
from psycopg_pool import PoolTimeout, AsyncConnectionPool


class DbConnector:
    """
    Connects to the given postgres database
    """

    async_connection_pool: AsyncConnectionPool = None

    async def init_pool(self, dbname: str, user: str, password: str, host: str):
        try:
            self.async_connection_pool = AsyncConnectionPool(
                open=False,
                conninfo=f"dbname={dbname} user={user} password={password} host={host}",
            )
            await self.async_connection_pool.open()
        except PoolTimeout as e:
            logging.critical(
                f"Could not initialize the connection pool for the PostgreSQL database! {e}"
            )

    async def close_pool(self):
        return await self.async_connection_pool.close()


class TradingdataRepository(DbConnector):
    use_cache: bool

    def __init__(self, use_cache=True):
        # Disable the cache in tests
        self.use_cache = False if "pytest" in sys.modules else use_cache

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

    def bulk_insert(self,
                    path=Path(__file__).parent / f"./data/data-{Config.get_value('TRADED_TICKER_NAME')}.csv") -> None:
        df: pd.DataFrame = self.load_ticker_data(path)
        engine = create_engine(
            f"postgresql+psycopg://{Config.get_value('POSTGRES_USER')}:{Config.get_value('POSTGRES_PASSWORD')}@{Config.get_value('POSTGRES_HOST')}:{Config.get_value('POSTGRES_PORT')}/{Config.get_value('POSTGRES_DB')}")
        df.to_sql(
            name="tradingdata",
            con=engine,
            if_exists='replace',
            index=True,
            chunksize=1000
        )

    async def get_latest_record(self) -> np.array:
        async with self.async_connection_pool.connection() as connection:
            async with connection.cursor() as curs:
                await curs.execute('select * from tradingdata order by "Date" Desc limit 1;')
                return await curs.fetchone()

import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from src.tradingbot.config import Config
from psycopg_pool import PoolTimeout, AsyncConnectionPool

from src.tradingbot.exceptions import BadTradingWeightsException


class DbConnector:
    """
    Connects to the given postgres database
    """

    async_connection_pool: AsyncConnectionPool = None
    db_name: str

    async def init_pool(self, dbname: str, user: str, password: str, host: str):
        self.db_name = dbname
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
    table_name: str

    def __init__(self, use_cache=True, table_name='tradingdata', db_name=Config.get_value('POSTGRES_DB')):
        # Disable the cache in tests
        self.use_cache = False if "pytest" in sys.modules else use_cache
        self.table_name = table_name
        self.db_name = db_name

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
            f"postgresql+psycopg://{Config.get_value('POSTGRES_USER')}:{Config.get_value('POSTGRES_PASSWORD')}@{Config.get_value('POSTGRES_HOST')}:{Config.get_value('POSTGRES_PORT')}/{self.db_name}")
        df.to_sql(
            name=self.table_name,
            con=engine,
            if_exists='replace',
            index=True,
            chunksize=1000
        )

    async def get_latest_record(self) -> np.array:
        async with self.async_connection_pool.connection() as connection:
            async with connection.cursor() as curs:
                await curs.execute(f'select * from {self.table_name} order by "Date" Desc limit 1;')
                result = await curs.fetchall()
                return result[0]

    async def get_all_records(self) -> np.array:
        async with self.async_connection_pool.connection() as connection:
            async with connection.cursor() as curs:
                await curs.execute(f'select * from {self.table_name} order by "Date" Desc;')
                result = await curs.fetchall()
                return result

    async def get_db_columns(self) -> np.array:
        async with self.async_connection_pool.connection() as connection:
            async with connection.cursor() as curs:
                await curs.execute(
                    f"select column_name from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME = '{self.table_name}'")
                return [el[0] for el in await curs.fetchall()]


class TradingStrategyWeightsRepository(DbConnector):
    table_name: str

    def __init__(self, table_name='trading_weights', db_name=Config.get_value('POSTGRES_DB')):
        self.table_name = table_name
        self.db_name = db_name

    def load_strategy_weights_data(self, path=Path(__file__).parent / f"./data/weights.csv") -> pd.DataFrame:
        return pd.read_csv(path)

    def insert(self, path=Path(__file__).parent / f"./data/weights.csv") -> None:
        df: pd.DataFrame = self.load_strategy_weights_data(path)
        engine = create_engine(
            f"postgresql+psycopg://{Config.get_value('POSTGRES_USER')}:{Config.get_value('POSTGRES_PASSWORD')}@{Config.get_value('POSTGRES_HOST')}:{Config.get_value('POSTGRES_PORT')}/{self.db_name}")
        df.to_sql(
            name=self.table_name,
            con=engine,
            if_exists='replace',
            index=False,
            chunksize=1000
        )

    async def get_weights(self) -> np.array:
        async with self.async_connection_pool.connection() as connection:
            async with connection.cursor() as curs:
                await curs.execute(f'select * from {self.table_name};')
                result = await curs.fetchall()
                if len(result) > 1:
                    raise BadTradingWeightsException(message='More than one row of trading weights found in DB!')
                elif len(result) == 0 or result[0] is None:
                    raise BadTradingWeightsException(message='No trading weights found in DB!')
                return result[0]

    async def get_db_columns(self) -> np.array:
        async with self.async_connection_pool.connection() as connection:
            async with connection.cursor() as curs:
                await curs.execute(
                    f"select column_name from INFORMATION_SCHEMA.COLUMNS where TABLE_NAME = '{self.table_name}'")
                return [el[0] for el in await curs.fetchall()]

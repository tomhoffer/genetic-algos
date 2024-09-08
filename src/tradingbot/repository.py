import logging
import sys
from pathlib import Path
from typing import Literal, Dict
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from src.tradingbot.config import Config
from psycopg_pool import PoolTimeout, AsyncConnectionPool

from src.tradingbot.exceptions import BadTradingWeightsException, NoDataFoundException


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
    cached_data: Dict[str, pd.DataFrame]

    def __init__(self, use_cache=True, table_name='tradingdata', db_name=Config.get_value('POSTGRES_DB')):
        # Disable the cache in tests
        self.use_cache = False if "pytest" in sys.modules else use_cache
        self.cached_data = {}
        self.table_name = table_name
        self.db_name = db_name

    def _is_in_cache(self, date_range: str) -> bool:
        if not self.use_cache:
            return False
        if date_range in self.cached_data and not self.cached_data[date_range].empty:
            return True
        return False

    def load_ticker_data(
            self,
            path=Path(__file__).parent / f"./data/data-{Config.get_value('TRADED_TICKER_NAME')}.csv",
            start_date: str = None,
            end_date: str = None,
    ) -> pd.DataFrame:
        date_range = f"{start_date}-{end_date}"
        if self._is_in_cache(date_range=date_range):
            return self.cached_data[date_range]

        if start_date and end_date:
            data = pd.read_csv(path, parse_dates=['Date'], index_col=['Date'])[start_date:end_date]
        else:
            data = pd.read_csv(path, parse_dates=['Date'], index_col=['Date'])

        if self.use_cache:
            self.cached_data[date_range] = data

        return data

    def upload_from_df(self, df: pd.DataFrame, if_exists: Literal['fail', 'replace', 'append'] = 'replace'):
        engine = create_engine(
            f"postgresql+psycopg://{Config.get_value('POSTGRES_USER')}:{Config.get_value('POSTGRES_PASSWORD')}@{Config.get_value('POSTGRES_HOST')}:{Config.get_value('POSTGRES_PORT')}/{self.db_name}")
        df.to_sql(
            name=self.table_name,
            con=engine,
            if_exists=if_exists,
            index=True,
            chunksize=1000
        )

    def upload_from_csv(self,
                        path=Path(__file__).parent / f"./data/data-{Config.get_value('TRADED_TICKER_NAME')}.csv",
                        if_exists: Literal['fail', 'replace', 'append'] = 'replace') -> None:
        df: pd.DataFrame = self.load_ticker_data(path)
        self.upload_from_df(df=df, if_exists=if_exists)

    async def get_latest_record(self) -> np.array:
        async with self.async_connection_pool.connection() as connection:
            async with connection.cursor() as curs:
                await curs.execute(f'select * from {self.table_name} order by "Date" Desc limit 1;')
                result = await curs.fetchall()
                try:
                    return result[0]
                except IndexError:
                    raise NoDataFoundException('No trading data found in the DB.')

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

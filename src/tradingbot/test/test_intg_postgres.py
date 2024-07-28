import uuid
from datetime import datetime

import pandas as pd
import pytest
import pytest_asyncio
import psycopg
from src.tradingbot.config import Config
from src.tradingbot.exceptions import BadTradingWeightsException
from src.tradingbot.repository import TradingdataRepository, TradingStrategyWeightsRepository

# Configuration for PostgreSQL connection
DB_CONFIG = {
    'user': Config.get_value('POSTGRES_USER'),
    'password': Config.get_value('POSTGRES_PASSWORD'),
    'host': Config.get_value('POSTGRES_HOST'),
    'port': Config.get_value('POSTGRES_PORT'),
    'dbname': 'trading_db'  # Default DB
}


@pytest.fixture(scope='function')
def unique_test_db_name():
    """Fixture to generate a unique test database name."""
    return f"test_db_{uuid.uuid4().hex}"


@pytest.fixture(scope='module')
def db_connection():
    """Fixture for establishing database connection."""
    conn = psycopg.connect(**DB_CONFIG)
    conn.set_autocommit(True)
    yield conn
    conn.close()


@pytest.fixture(scope='function')
def db_cursor(db_connection):
    """Fixture for providing a cursor to interact with the database."""
    with db_connection.cursor() as cur:
        yield cur


@pytest.fixture(scope='function')
def create_db(db_cursor, unique_test_db_name):
    """Fixture for creating the testing database."""
    create_db_query = f"CREATE DATABASE {unique_test_db_name}"
    db_cursor.execute(create_db_query)


@pytest_asyncio.fixture(scope='function')
async def tradingdata_repository(create_db, unique_test_db_name) -> TradingdataRepository:
    repository = TradingdataRepository(db_name=unique_test_db_name)
    await repository.init_pool(user=DB_CONFIG['user'], password=DB_CONFIG['password'],
                               dbname=unique_test_db_name,
                               host=DB_CONFIG['host'])
    yield repository


@pytest_asyncio.fixture(scope='function')
async def tradingweights_repository(create_db, unique_test_db_name) -> TradingStrategyWeightsRepository:
    repository = TradingStrategyWeightsRepository(db_name=unique_test_db_name)
    await repository.init_pool(user=DB_CONFIG['user'], password=DB_CONFIG['password'],
                               dbname=unique_test_db_name,
                               host=DB_CONFIG['host'])
    yield repository


@pytest_asyncio.fixture(scope='function')
async def drop_db(db_cursor, create_db, unique_test_db_name, tradingdata_repository, tradingweights_repository):
    """Fixture to drop a temporary database for testing."""
    yield unique_test_db_name
    await tradingdata_repository.close_pool()
    drop_db_query = f"DROP DATABASE {unique_test_db_name} WITH (FORCE);"
    db_cursor.execute(drop_db_query)


@pytest.mark.asyncio
async def test_tradingdata_bulk_insert(tradingdata_repository: TradingdataRepository, drop_db):
    df: pd.DataFrame = tradingdata_repository.load_ticker_data()
    tradingdata_repository.bulk_insert()
    result = await tradingdata_repository.get_all_records()
    assert len(result) == df.shape[0]


@pytest.mark.asyncio
async def test_column_order(tradingdata_repository: TradingdataRepository, drop_db):
    # Test if order of columns in the uploaded df is preserved in the DB
    df: pd.DataFrame = tradingdata_repository.load_ticker_data('src/tradingbot/test/test_data.csv')
    tradingdata_repository.bulk_insert('src/tradingbot/test/test_data.csv')
    result = await tradingdata_repository.get_db_columns()
    assert result == [df.index.name] + df.columns.tolist()


@pytest.mark.asyncio
async def test_tradingdata_get_latest_record(tradingdata_repository: TradingdataRepository, drop_db):
    tradingdata_repository.bulk_insert('src/tradingbot/test/test_data.csv')
    result = await tradingdata_repository.get_latest_record()
    assert result == (datetime(1970, 1, 10, 0, 0), 10, 1, 1.9)


@pytest.mark.asyncio
async def test_tradingdata_get_all_records(tradingdata_repository: TradingdataRepository, drop_db):
    tradingdata_repository.bulk_insert('src/tradingbot/test/test_data.csv')
    result = await tradingdata_repository.get_all_records()
    assert result == [(datetime(1970, 1, 10, 0, 0), 10, 1, 1.9),
                      (datetime(1970, 1, 9, 0, 0), 9, 2, 1.8),
                      (datetime(1970, 1, 8, 0, 0), 8, 3, 1.7),
                      (datetime(1970, 1, 7, 0, 0), 7, 4, 1.6),
                      (datetime(1970, 1, 6, 0, 0), 6, 5, 1.5),
                      (datetime(1970, 1, 5, 0, 0), 5, 6, 1.4),
                      (datetime(1970, 1, 4, 0, 0), 4, 7, 1.3),
                      (datetime(1970, 1, 3, 0, 0), 3, 8, 1.2),
                      (datetime(1970, 1, 2, 0, 0), 2, 9, 1.1),
                      (datetime(1970, 1, 1, 0, 0), 1, 10, 1.0)]


@pytest.mark.asyncio
async def test_tradingweights_insert(tradingweights_repository: TradingStrategyWeightsRepository, drop_db):
    df: pd.DataFrame = tradingweights_repository.load_strategy_weights_data('src/tradingbot/test/weights_data.csv')
    tradingweights_repository.insert('src/tradingbot/test/weights_data.csv')

    # Test that data has been properly inserted
    result = list(await tradingweights_repository.get_weights())
    assert list(result) == df.values.tolist()[0]

    # Test that second insert overwrites previous data
    tradingweights_repository.insert('src/tradingbot/test/weights_data.csv')
    result = list(await tradingweights_repository.get_weights())
    assert list(result) == df.values.tolist()[0]


@pytest.mark.asyncio
async def test_tradingweights_get_weights_invalid(mocker, tradingweights_repository: TradingStrategyWeightsRepository,
                                                  drop_db):
    # Test that BadTradingWeightsException is raised for no rows
    df: pd.DataFrame = pd.DataFrame(data={'strategy1': [], 'strategy2': []})
    mocker.patch(
        "src.tradingbot.repository.TradingStrategyWeightsRepository.load_strategy_weights_data", return_value=df
    )
    tradingweights_repository.insert()
    with pytest.raises(BadTradingWeightsException, match='No trading weights found in DB!'):
        await tradingweights_repository.get_weights()

    # Test that BadTradingWeightsException is raised for multiple rows (Multiple weights are not allowed)
    df: pd.DataFrame = pd.DataFrame(data={'strategy1': [1, 2], 'strategy2': [3, 4]})
    mocker.patch(
        "src.tradingbot.repository.TradingStrategyWeightsRepository.load_strategy_weights_data", return_value=df
    )
    tradingweights_repository.insert()
    with pytest.raises(BadTradingWeightsException, match='More than one row of trading weights found in DB!'):
        await tradingweights_repository.get_weights()


if __name__ == "__main__":
    pytest.main([__file__])

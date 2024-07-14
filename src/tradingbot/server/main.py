import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.tradingbot.repository import TradingdataRepository
from src.tradingbot.server.config import Config

trading_db = TradingdataRepository()
logger = logging.getLogger('uvicorn.error')


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug(os.environ)
    await trading_db.init_pool(
        dbname=Config.get_value("POSTGRES_DB"),
        user=Config.get_value("POSTGRES_DB_USER"),
        password=Config.get_value("POSTGRES_DB_PASSWORD"),
        host=Config.get_value("POSTGRES_DB_HOST"),
    )
    yield
    await trading_db.close_pool()


app = FastAPI(lifespan=lifespan)


@app.get("/decision/")
async def read_item():
    record = await trading_db.get_latest_record()
    return record
    """
    # Make decision based on all trading strategies and their weights

    ts = TradingStrategies()
    decisions: np.array = np.array(list(ts.perform_decisions_for_row(row, row_np_index).values()))
    decisions_sum = np.sum(np.multiply(solution.parse_strategy_weights(), decisions))
    """

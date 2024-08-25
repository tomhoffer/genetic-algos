import logging
import os
from contextlib import asynccontextmanager
from typing import List

import numpy as np
from fastapi import FastAPI

from src.tradingbot.decisions import TradingStrategies
from src.tradingbot.enums import Decision
from src.tradingbot.repository import TradingdataRepository, TradingStrategyWeightsRepository
from src.tradingbot.server.config import Config

trading_data_repository = TradingdataRepository()
trading_weights_repository = TradingStrategyWeightsRepository()
logger = logging.getLogger('uvicorn.error')


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.debug(os.environ)
    await trading_data_repository.init_pool(
        dbname=Config.get_value("POSTGRES_DB"),
        user=Config.get_value("POSTGRES_DB_USER"),
        password=Config.get_value("POSTGRES_DB_PASSWORD"),
        host=Config.get_value("POSTGRES_DB_HOST"),
    )
    await trading_weights_repository.init_pool(
        dbname=Config.get_value("POSTGRES_DB"),
        user=Config.get_value("POSTGRES_DB_USER"),
        password=Config.get_value("POSTGRES_DB_PASSWORD"),
        host=Config.get_value("POSTGRES_DB_HOST"),
    )
    yield
    await trading_data_repository.close_pool()
    await trading_weights_repository.close_pool()


app = FastAPI(lifespan=lifespan)


@app.get("/decision/")
async def get_decision():
    latest_trading_indicators: np.ndarray = await trading_data_repository.get_latest_record()
    trading_indicator_names: List[str] = await trading_data_repository.get_db_columns()
    strategy_weights: np.ndarray = await trading_weights_repository.get_weights()

    # Make decision based on all trading strategies and their weights
    ts = TradingStrategies()
    row_np_index = dict(zip(trading_indicator_names, list(range(0, len(trading_indicator_names)))))
    decisions: np.array = np.array(
        list(ts.perform_decisions_for_row(np.array(latest_trading_indicators), row_np_index).values()))
    decisions_sum = np.sum(np.multiply(strategy_weights, decisions))

    if decisions_sum > 0:
        final_decision = Decision.BUY
    elif decisions_sum < 0:
        final_decision = Decision.SELL
    else:
        final_decision = Decision.INCONCLUSIVE
    return {
        "decision": final_decision
    }

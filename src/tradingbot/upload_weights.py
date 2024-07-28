import asyncio

from src.tradingbot.config import Config
from src.tradingbot.repository import TradingStrategyWeightsRepository


async def main():
    trading_weights_repository = TradingStrategyWeightsRepository()
    await trading_weights_repository.init_pool(
        dbname=Config.get_value("POSTGRES_DB"),
        user=Config.get_value("POSTGRES_USER"),
        password=Config.get_value("POSTGRES_PASSWORD"),
        host=Config.get_value("POSTGRES_HOST"),
    )
    trading_weights_repository.load_strategy_weights_data('data/weights.csv')
    trading_weights_repository.insert('data/weights.csv')
    await trading_weights_repository.close_pool()


loop = asyncio.get_event_loop()
loop.run_until_complete(main())

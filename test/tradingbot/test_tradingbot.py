from typing import List
import numpy as np
from conftest import mockenv

from src.tradingbot.decisions import Decision
from src.tradingbot.tradingbot import TradingbotSolution, initial_population_generator, Position, load_ticker_data, \
    fitness


@mockenv(POPULATION_SIZE="10")
def test_generate_initial_population():
    population: List[TradingbotSolution] = initial_population_generator()

    assert len(population) == 10
    for element in population:
        # Timestamps within range
        assert np.all((element.chromosome >= 0) & (element.chromosome <= 1))


def test_tradingbotsolution_sell_all(mocker):
    mocked_sell = mocker.patch('src.tradingbot.tradingbot.TradingbotSolution.sell')
    s = TradingbotSolution(chromosome=np.asarray([]))
    s.bought_positions = [Position(datetime="1", amount=1) for _ in range(10)]
    s.sell_all(datetime="1")
    assert mocked_sell.call_count == 10


def test_tradingbotsolution_sell_all_bought_positions(mocker):
    df = load_ticker_data('test/tradingbot/test_data.csv')
    mocker.patch('src.tradingbot.tradingbot.load_ticker_data', return_value=df)
    s = TradingbotSolution(chromosome=np.asarray([]))
    s.bought_positions = [Position(datetime='1970-01-01', amount=1) for _ in range(10)]
    s.sell_all(datetime='1970-01-01')
    assert len(s.bought_positions) == 0


@mockenv(TRADED_TICKER_NAME="AAPL")
def test_tradingbotsolution_sell(mocker):
    df = load_ticker_data('test/tradingbot/test_data.csv')
    mocker.patch('src.tradingbot.tradingbot.load_ticker_data', return_value=df)
    s = TradingbotSolution(chromosome=np.asarray([]))
    timestamp_bought = '1970-01-01'
    timestamp_sold = '1970-01-10'

    s.bought_positions = [Position(datetime=timestamp_bought, amount=1) for _ in range(10)]
    s.account_balance = 0

    s.sell(datetime=timestamp_sold)

    assert len(s.bought_positions) == 9
    assert s.account_balance == 9


@mockenv(TRADED_TICKER_NAME="AAPL", START_TIMESTAMP="34361", END_TIMESTAMP="811961", BUDGET="100", TRADE_SIZE="10")
def test_tradingbotsolution_fitness(mocker):
    # Fitness works correctly for a dummy chromosome with only 1 strategy (100% weight)
    # Date range = 1970-01-01 -> 1970-01-10
    df = load_ticker_data('test/tradingbot/test_data.csv')
    mocker.patch('src.tradingbot.tradingbot.load_ticker_data', return_value=df)
    chromosome = np.asarray([1.0])

    # Decisions: Buy gradually every day, sell everything at the end
    mocker.patch('src.tradingbot.decisions.TradingStrategies.perform_decisions_for_row',
                 return_value={"dummy_strategy": Decision.BUY})
    result: float = fitness(chromosome)
    np.testing.assert_almost_equal(result, 192.89, decimal=2)

    # Decisions: Inconclusive every day
    mocker.patch('src.tradingbot.decisions.TradingStrategies.perform_decisions_for_row',
                 return_value={"dummy_strategy": Decision.INCONCLUSIVE})
    result: float = fitness(chromosome)
    np.testing.assert_almost_equal(result, 100, decimal=2)

    # Decisions: Sell every day (nothing to sell)
    mocker.patch('src.tradingbot.decisions.TradingStrategies.perform_decisions_for_row',
                 return_value={"dummy_strategy": Decision.SELL})
    result: float = fitness(chromosome)
    np.testing.assert_almost_equal(result, 100, decimal=2)

    # Decisions: Buy in days 1-5, Sell in days 6-10
    mocker.patch('src.tradingbot.decisions.TradingStrategies.perform_decisions_for_row',
                 side_effect=[{"dummy_strategy": Decision.BUY if i < 5 else Decision.SELL} for i in range(10)])
    result: float = fitness(chromosome)
    np.testing.assert_almost_equal(result, 164.16, decimal=2)
from typing import List
import numpy as np
import pytest
from conftest import mockenv
from src.tradingbot.decisions import Decision
from src.tradingbot.exceptions import InvalidTradeActionException
from src.tradingbot.tradingbot import TradingbotSolution, initial_population_generator, BuyPosition, load_ticker_data, \
    fitness, chromosome_validator_fn


@mockenv(POPULATION_SIZE="10")
def test_generate_initial_population():
    population: List[TradingbotSolution] = initial_population_generator()

    assert len(population) == 10
    for element in population:
        assert chromosome_validator_fn(element)


def test_tradingbotsolution_sell_all(mocker):
    mocked_sell = mocker.patch('src.tradingbot.tradingbot.TradingbotSolution.sell')
    s = TradingbotSolution(chromosome=np.asarray([]))
    s.bought_positions = [BuyPosition(datetime="1", amount=1, price_at_buy=1) for _ in range(10)]
    s.sell_all(datetime="1")
    assert mocked_sell.call_count == 10


def test_tradingbotsolution_sell_all_bought_positions(mocker):
    df = load_ticker_data('test/tradingbot/test_data.csv')
    mocker.patch('src.tradingbot.tradingbot.load_ticker_data', return_value=df)
    s = TradingbotSolution(chromosome=np.asarray([]))
    s.bought_positions = [BuyPosition(datetime='1970-01-01', amount=1, price_at_buy=1) for _ in range(10)]
    s.sell_all(datetime='1970-01-01')
    assert len(s.bought_positions) == 0


@mockenv(TRADED_TICKER_NAME="AAPL")
def test_tradingbotsolution_sell(mocker):
    df = load_ticker_data('test/tradingbot/test_data.csv')
    mocker.patch('src.tradingbot.tradingbot.load_ticker_data', return_value=df)
    s = TradingbotSolution(chromosome=np.asarray([]))
    timestamp_bought = '1970-01-01'
    timestamp_sold = '1970-01-10'

    s.bought_positions = [BuyPosition(datetime=timestamp_bought, amount=1, price_at_buy=1) for _ in range(10)]
    s.account_balance = 0

    s.sell(datetime=timestamp_sold, index=0)

    assert len(s.bought_positions) == 9
    assert s.account_balance == 1 + 9


@mockenv(TRADED_TICKER_NAME="AAPL", BUDGET="20", TRADE_SIZE="10")
def test_buy_sell(mocker):
    df = load_ticker_data('test/tradingbot/test_data.csv')
    mocker.patch('src.tradingbot.tradingbot.load_ticker_data', return_value=df)
    s = TradingbotSolution(chromosome=np.asarray([]), account_balance=20)
    timestamp_bought = '1970-01-01'
    timestamp_sold = '1970-01-05'

    s.buy(datetime=timestamp_bought, amount=10, price=1)
    assert s.account_balance == 10

    s.sell(datetime=timestamp_sold, index=0)  # Profit 5x

    assert len(s.bought_positions) == 0
    assert s.account_balance == 10 + 50


@mockenv(TRADED_TICKER_NAME="INCREASING", START_TIMESTAMP="34361", END_TIMESTAMP="811961", BUDGET="10", TRADE_SIZE="10")
def test_take_profit(mocker):
    # Buy on first date, inconclusive on other dates, Take profit should trigger when threshold is hit
    df = load_ticker_data('test/tradingbot/test_data.csv')
    mocker.patch('src.tradingbot.tradingbot.load_ticker_data', return_value=df)
    mocker.patch('src.tradingbot.decisions.TradingStrategies.perform_decisions_for_row',
                 side_effect=[{"dummy_strategy": Decision.BUY if i == 0 else Decision.INCONCLUSIVE} for i in range(10)])

    chromosome = np.asarray([1.0, 1.2, 1.0])
    s = TradingbotSolution(chromosome=chromosome)
    s.buy(datetime='1970-01-01', price=10, amount=1)
    result = fitness(chromosome)
    np.testing.assert_almost_equal(result, 12, decimal=2)


@mockenv(TRADED_TICKER_NAME="DECREASING", START_TIMESTAMP="34361", END_TIMESTAMP="811961", BUDGET="10", TRADE_SIZE="10")
def test_stop_loss(mocker):
    # Buy on first date, inconclusive on other dates, Stop loss should trigger when threshold is hit
    df = load_ticker_data('test/tradingbot/test_data.csv')
    mocker.patch('src.tradingbot.tradingbot.load_ticker_data', return_value=df)
    mocker.patch('src.tradingbot.decisions.TradingStrategies.perform_decisions_for_row',
                 side_effect=[{"dummy_strategy": Decision.BUY if i == 0 else Decision.INCONCLUSIVE} for i in range(10)])

    chromosome = np.asarray([1.0, 2.0, 0.8])
    s = TradingbotSolution(chromosome=chromosome)
    s.buy(datetime='1970-01-01', price=10, amount=1)
    result = fitness(chromosome)
    np.testing.assert_almost_equal(result, 8, decimal=2)


def test_sell_invalid_index(mocker):
    # Sell a position on an index out of range
    df = load_ticker_data('test/tradingbot/test_data.csv')
    mocker.patch('src.tradingbot.tradingbot.load_ticker_data', return_value=df)
    s = TradingbotSolution(chromosome=np.asarray([]))
    s.bought_positions = [BuyPosition(datetime='1970-01-01', amount=1, price_at_buy=1) for _ in range(10)]
    with pytest.raises(InvalidTradeActionException):
        s.sell(datetime='1970-01-01', index=10)


@mockenv(TRADED_TICKER_NAME="AAPL", BUDGET="20", TRADE_SIZE="10")
def test_buy_partial(mocker):
    # Buy is possible for a partial amount if account balance > 0
    df = load_ticker_data('test/tradingbot/test_data.csv')
    mocker.patch('src.tradingbot.tradingbot.load_ticker_data', return_value=df)
    s = TradingbotSolution(chromosome=np.asarray([]), account_balance=20)
    timestamp_bought = '1970-01-01'

    s.buy(datetime=timestamp_bought, amount=30, price=1)
    assert s.account_balance == 0
    assert len(s.bought_positions) == 1
    assert s.bought_positions[0].amount == 20


@mockenv(TRADED_TICKER_NAME="AAPL", BUDGET="20", TRADE_SIZE="10")
def test_buy_failed(mocker):
    df = load_ticker_data('test/tradingbot/test_data.csv')
    mocker.patch('src.tradingbot.tradingbot.load_ticker_data', return_value=df)

    # Buy is not possible when account balance is 0
    s = TradingbotSolution(chromosome=np.asarray([]), account_balance=0)
    timestamp_bought = '1970-01-01'
    s.buy(datetime=timestamp_bought, amount=1, price=1)
    assert s.account_balance == 0
    assert len(s.bought_positions) == 0


@mockenv(TRADED_TICKER_NAME="AAPL", START_TIMESTAMP="34361", END_TIMESTAMP="811961", BUDGET="100", TRADE_SIZE="10")
def test_tradingbotsolution_fitness(mocker):
    # Fitness works correctly for a dummy chromosome with only 1 strategy (100% weight)
    # Date range = 1970-01-01 -> 1970-01-10
    # Stop loss and take profit are intentionally suppressed in this test
    df = load_ticker_data('test/tradingbot/test_data.csv')
    mocker.patch('src.tradingbot.tradingbot.load_ticker_data', return_value=df)
    chromosome = np.asarray([1.0, 1000.0, 0])

    # Decisions: Buy gradually every day, sell everything at the end
    mocker.patch('src.tradingbot.decisions.TradingStrategies.perform_decisions_for_row',
                 return_value={"dummy_strategy": Decision.BUY})
    result: float = fitness(chromosome)
    np.testing.assert_almost_equal(result, 292.89, decimal=2)

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
    np.testing.assert_almost_equal(result, 214.16, decimal=2)

from typing import List
import numpy as np
import pandas as pd
import pytest

from src.investobot.investobot import initial_population_generator, InvestobotSolution, mutate, crossover, fitness
from conftest import mockenv


@mockenv(BUDGET="1000", CHROMOSOME_MAX_LENGTH="15", START_TIMESTAMP="0", END_TIMESTAMP="1000", POPULATION_SIZE="10")
def test_generate_initial_population(mocker):
    mocker.patch("src.investobot.investobot.create_ticker_list", return_value=['AAPL', 'GOOG', 'CNDX'])
    population: List[InvestobotSolution] = initial_population_generator()

    assert len(population) == 10
    for element in population:
        timestamps = element.get_chromosome_timestamps()
        tickers = element.get_chromosome_tickers()
        amounts = element.get_chromosome_amounts()

        # Maximum investments limit
        assert element.chromosome.shape[0] <= 15

        # Timestamps within range
        assert np.all((timestamps >= 0) & (timestamps <= 1000))

        # Valid ticker ids
        assert np.all((tickers >= 0) & (tickers < 3))

        # Budget matches up
        np.testing.assert_almost_equal(amounts.sum(), 1000, decimal=5)


@mockenv(BUDGET="1000", START_TIMESTAMP="0", END_TIMESTAMP="1000", P_MUTATION="1.0")
def test_mutate(mocker):
    mocker.patch("src.investobot.investobot.create_ticker_list", return_value=['AAPL', 'GOOG', 'CNDX', 'SPY'])
    before = np.asarray([
        [0, 250.0, 1],
        [1, 250.0, 250],
        [2, 250.0, 500],
        [3, 250.0, 1000],
    ])
    after = mutate(before)
    assert before.shape == after.shape

    after_timestamps = InvestobotSolution.parse_chromosome_timestamps(after)
    after_amounts = InvestobotSolution.parse_chromosome_amounts(after)
    after_tickers = InvestobotSolution.parse_chromosome_tickers(after)

    assert np.all((after_timestamps >= 0) & (after_timestamps <= 1000))
    assert np.all((after_tickers >= 0) & (after_tickers <= 3))
    np.testing.assert_almost_equal(after_amounts.sum(), 1000, decimal=5)


@mockenv(BUDGET="1000")
def test_crossover():
    parent1 = InvestobotSolution(np.asarray([
        [0, 250.0, 1],
        [1, 250.0, 250],
        [2, 250.0, 500],
        [3, 250.0, 1000],
    ]))
    parent2 = InvestobotSolution(np.asarray([
        [1, 100.0, 1],
        [1, 400.0, 100],
        [3, 250.0, 500],
        [2, 250.0, 700],
    ]))
    res1, res2 = crossover(parent1, parent2)

    assert res1.chromosome.shape == res2.chromosome.shape == parent1.chromosome.shape == parent2.chromosome.shape

    unique_parent_tickers = set(parent1.get_chromosome_tickers() + parent2.get_chromosome_tickers())
    unique_result_tickers = set(res1.get_chromosome_tickers() + res2.get_chromosome_tickers())
    unique_parent_timestamps = set(parent1.get_chromosome_timestamps() + parent2.get_chromosome_timestamps())
    unique_result_timestamps = set(res1.get_chromosome_timestamps() + res2.get_chromosome_timestamps())

    # Tickers and timestamps come from parent chromosomes
    assert unique_parent_timestamps == unique_result_timestamps
    assert unique_parent_tickers == unique_result_tickers

    # Amounts are adjusted to match the budget
    np.testing.assert_almost_equal(res1.get_chromosome_amounts().sum(), 1000, decimal=5)
    np.testing.assert_almost_equal(res2.get_chromosome_amounts().sum(), 1000, decimal=5)


@mockenv(END_TIMESTAMP="792000", BUDGET="10")  # 1970-01-10
@pytest.mark.parametrize("ticker_values, invested_ticker, expected_fitness",
                         [
                             (np.linspace(1, 10, num=10), 0, 90),  # Growing stock value
                             (np.linspace(10, 1, num=10), 0, -90),  # Decreasing stock value
                             (np.full(10, 10), 0, 0),  # Constant stock value
                             (np.full(10, np.nan), 0, np.NINF),  # Missing all stock values
                             (np.asarray([1, 2, 3, 4, 5, np.nan, 7, 8, 9, 10]), 0, 90),
                             # Missing stock value in between has no impact
                             (np.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9, np.nan]), 0, np.NINF),
                             # Missing stock value at END_TIMESTAMP
                             (np.linspace(1, 10, num=10), 1, np.NINF),
                             # Missing stock column (stock with id=1 does not exist)
                         ])
def test_fitness(ticker_values, invested_ticker, expected_fitness, mocker):
    dates = [f'1970-01-0{i}' if i < 10 else f'1970-01-{i}' for i in range(1, 11)]
    data = {'Date': dates, 'AAPL': list(ticker_values)}
    df = pd.DataFrame(data=data)
    df = df.set_index('Date')
    mocker.patch('src.investobot.investobot.load_tickers', return_value=df)

    # Investing all budget on given ticker on the first date (all-in)
    chromosome = np.asarray([[invested_ticker, 10, 0]])

    result = fitness(chromosome)
    np.testing.assert_almost_equal(result, expected_fitness, decimal=5)

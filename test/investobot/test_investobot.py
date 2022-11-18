from typing import List
import numpy as np
from src.investobot.investobot import initial_population_generator, InvestobotSolution
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
        assert np.all((timestamps > 0) & (timestamps < 1000))

        # Valid ticker ids
        assert np.all((tickers >= 0) & (tickers < 3))

        # Budget matches up
        np.testing.assert_almost_equal(amounts.sum(), 1000, decimal=5)

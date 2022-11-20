import logging
import os
from functools import cache
from random import randint
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numpy.random import default_rng

from src.generic.model import Solution
from src.generic.mutation import Mutation

logging.basicConfig(level=logging.INFO)
load_dotenv()


class InvestobotSolution(Solution):
    def get_chromosome_tickers(self) -> np.ndarray:
        return self.parse_chromosome_tickers(self.chromosome)

    def get_chromosome_amounts(self) -> np.ndarray:
        return self.parse_chromosome_amounts(self.chromosome)

    def get_chromosome_timestamps(self) -> np.ndarray:
        return self.parse_chromosome_timestamps(self.chromosome)

    @staticmethod
    def parse_chromosome_tickers(chromosome: np.ndarray) -> np.ndarray:
        return chromosome[:, 0]

    @staticmethod
    def parse_chromosome_amounts(chromosome: np.ndarray) -> np.ndarray:
        return chromosome[:, 1]

    @staticmethod
    def parse_chromosome_timestamps(chromosome: np.ndarray) -> np.ndarray:
        return chromosome[:, 2]


@cache
def create_ticker_list() -> List[str]:
    df: pd.DataFrame = pd.read_csv('data.csv')
    tickers: List[str] = list(df.columns)
    return tickers[1:]


def initial_population_generator() -> List[InvestobotSolution]:
    result: List[InvestobotSolution] = []

    max_transactions: int = int(os.environ.get("CHROMOSOME_MAX_LENGTH"))
    ticker_list: List[str] = create_ticker_list()
    budget = float(os.environ.get("BUDGET"))
    min_timestamp = int(os.environ.get("START_TIMESTAMP"))
    max_timestamp = int(os.environ.get("END_TIMESTAMP"))

    rng = default_rng()
    for _ in range(int(os.environ.get("POPULATION_SIZE"))):

        number_of_investments = randint(1, max_transactions)

        # Proportions of the budget assigned to invest in each asset
        proportions = rng.dirichlet(np.ones(number_of_investments)).flatten()

        chromosome = np.empty((number_of_investments, 3))
        for i in range(number_of_investments):
            ticker_id = randint(0, len(ticker_list) - 1)
            timestamp = randint(min_timestamp, max_timestamp)
            amount = proportions[i] * budget
            chromosome[i] = [ticker_id, amount, timestamp]

        result.append(InvestobotSolution(chromosome=chromosome))
    return result


def fitness(chromosome: np.ndarray) -> float:
    yield


def stopping_criteria_fn(solution: Solution) -> bool:
    yield


def mutate(chromosome: np.ndarray) -> np.ndarray:
    ticker_list: List[str] = create_ticker_list()
    tickers: np.ndarray = InvestobotSolution.parse_chromosome_tickers(chromosome)
    amounts: np.ndarray = InvestobotSolution.parse_chromosome_amounts(chromosome)
    timestamps: np.ndarray = InvestobotSolution.parse_chromosome_timestamps(chromosome)

    mutated_tickers = Mutation.mutate_real_uniform(tickers, min=0, max=len(ticker_list) - 1)
    mutated_amounts = Mutation.mutate_real_gaussian(amounts, use_abs=True)
    mutated_timestamps = Mutation.mutate_real_gaussian(timestamps, min=0, max=int(os.environ.get("END_TIMESTAMP")))

    # Ensure budget is fully used and not exceeded
    mutated_amounts_diff = mutated_amounts.sum() - float(os.environ.get("BUDGET"))
    mutated_amounts_diff_per_ticker = abs(mutated_amounts_diff / len(tickers))

    if mutated_amounts_diff > 0:
        mutated_amounts = mutated_amounts - mutated_amounts_diff_per_ticker

    if mutated_amounts_diff < 0:
        mutated_amounts = mutated_amounts + mutated_amounts_diff_per_ticker

    return np.stack((mutated_tickers, mutated_amounts, mutated_timestamps), axis=-1)


def crossover():
    yield


if __name__ == "__main__":
    initial_population_generator()

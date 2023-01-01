import logging
import os
from functools import cache
from random import randint
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from numpy.random import default_rng

from src.generic.crossover import Crossover
from src.generic.executor import TrainingExecutor
from src.generic.model import Solution, Hyperparams
from src.generic.mutation import Mutation
from src.generic.selection import Selection

logging.basicConfig(level=logging.INFO)
load_dotenv()


def post_process_amounts(amounts: np.ndarray):
    # Ensure budget is fully used and not exceeded
    amounts_diff = amounts.sum() - float(os.environ.get("BUDGET"))
    amounts_diff_per_ticker = abs(amounts_diff / len(amounts))
    updated_amounts = amounts
    if amounts_diff > 0:
        updated_amounts = amounts - amounts_diff_per_ticker

    if amounts_diff < 0:
        updated_amounts = amounts + amounts_diff_per_ticker

    return updated_amounts


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
    df = load_tickers()
    return list(df.columns)


@cache
def load_tickers(path: str = 'data.csv') -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'], index_col=['Date']).dropna(how='all', axis='columns')
    return df


def initial_population_generator() -> List[InvestobotSolution]:
    result: List[InvestobotSolution] = []

    num_transactions: int = int(os.environ.get("NUM_TRANSACTIONS"))
    ticker_list: List[str] = create_ticker_list()
    budget = float(os.environ.get("BUDGET"))
    min_timestamp = int(os.environ.get("START_TIMESTAMP"))
    max_timestamp = int(os.environ.get("END_TIMESTAMP"))

    rng = default_rng()
    for _ in range(int(os.environ.get("POPULATION_SIZE"))):

        # Proportions of the budget assigned to invest in each asset
        proportions = rng.dirichlet(np.ones(num_transactions)).flatten()

        chromosome = np.empty((num_transactions, 3))
        for i in range(num_transactions):
            ticker_id = randint(0, len(ticker_list) - 1)
            timestamp = randint(min_timestamp, max_timestamp)
            amount = proportions[i] * budget
            chromosome[i] = [ticker_id, amount, timestamp]

        result.append(InvestobotSolution(chromosome=chromosome))
    return result


def fitness(chromosome: np.ndarray) -> float:
    df = load_tickers()
    ticker_list = create_ticker_list()
    end_timestamp = int(os.environ.get("END_TIMESTAMP"))
    end_date: str = pd.to_datetime(end_timestamp, unit='s').strftime('%Y-%m-%d')

    def fitness_per_gene(row) -> float:
        try:
            ticker_name: str = ticker_list[int(row[0])]
        except IndexError:
            return np.nan

        invested_amount: float = row[1]
        invested_timestamp: int = row[2]
        invested_date: str = pd.to_datetime(invested_timestamp, unit='s').strftime('%Y-%m-%d')

        try:
            value_at_end: pd.Series = df.at[end_date, ticker_name]
        except KeyError:
            logging.error("Trading dataset does not contain data for given end date: " + end_date)
            raise KeyError

        try:
            value_at_invested: pd.Series = df.at[invested_date, ticker_name]
        except KeyError:
            # Start value is not present, unable to compute fitness
            return np.nan

        try:
            res = (value_at_end / value_at_invested) * invested_amount
            return res
        except ZeroDivisionError:
            logging.error(
                "Found ticker value equal to zero! Stopping fitness calculation to prevent division by zero...")

    res = np.apply_along_axis(fitness_per_gene, axis=1, arr=chromosome).sum()

    if np.isnan(res):
        # If market data is missing for any of given chromosomes
        return np.NINF
    return res


def stopping_criteria_fn(solution: Solution) -> bool:
    if solution.fitness > 1000000:
        return True
    else:
        return False


def mutate(chromosome: np.ndarray) -> np.ndarray:
    ticker_list: List[str] = create_ticker_list()
    tickers: np.ndarray = InvestobotSolution.parse_chromosome_tickers(chromosome)
    amounts: np.ndarray = InvestobotSolution.parse_chromosome_amounts(chromosome)
    timestamps: np.ndarray = InvestobotSolution.parse_chromosome_timestamps(chromosome)

    mutated_tickers = Mutation.mutate_real_uniform(tickers, min=0, max=len(ticker_list) - 1)
    mutated_amounts = Mutation.mutate_real_gaussian(amounts, use_abs=True)
    mutated_timestamps = Mutation.mutate_real_gaussian(timestamps, min=0, max=int(os.environ.get("END_TIMESTAMP")))

    # Ensure budget is fully used and not exceeded
    mutated_amounts = post_process_amounts(mutated_amounts)

    return np.stack((mutated_tickers, mutated_amounts, mutated_timestamps), axis=-1)


def crossover(parent1: InvestobotSolution, parent2: InvestobotSolution):
    res1, res2 = Crossover.two_point(parent1, parent2)

    res1_tickers: np.ndarray = InvestobotSolution.parse_chromosome_tickers(res1.chromosome)
    res1_amounts: np.ndarray = InvestobotSolution.parse_chromosome_amounts(res1.chromosome)
    res1_timestamps: np.ndarray = InvestobotSolution.parse_chromosome_timestamps(res1.chromosome)

    res2_tickers: np.ndarray = InvestobotSolution.parse_chromosome_tickers(res2.chromosome)
    res2_amounts: np.ndarray = InvestobotSolution.parse_chromosome_amounts(res2.chromosome)
    res2_timestamps: np.ndarray = InvestobotSolution.parse_chromosome_timestamps(res2.chromosome)

    # Ensure budget is fully used and not exceeded
    res1_updated_amounts = post_process_amounts(res1_amounts)
    res2_updated_amounts = post_process_amounts(res2_amounts)

    return (
        InvestobotSolution(chromosome=np.stack((res1_tickers, res1_updated_amounts, res1_timestamps), axis=-1)),
        InvestobotSolution(chromosome=np.stack((res2_tickers, res2_updated_amounts, res2_timestamps), axis=-1)),
    )


if __name__ == "__main__":
    params = Hyperparams(crossover_fn=crossover,
                         initial_population_generator_fn=initial_population_generator,
                         mutation_fn=mutate,
                         selection_fn=Selection.tournament,
                         fitness_fn=fitness, population_size=int(os.environ.get("POPULATION_SIZE")), elitism=5,
                         stopping_criteria_fn=stopping_criteria_fn)

    winner, success, id = TrainingExecutor.run((params, 1))
    winner_tickers: np.ndarray = InvestobotSolution.parse_chromosome_tickers(winner.chromosome)
    winner_amounts: np.ndarray = InvestobotSolution.parse_chromosome_amounts(winner.chromosome)
    winner_timestamps: np.ndarray = InvestobotSolution.parse_chromosome_timestamps(winner.chromosome)
    winner_timestamps_formatted = [pd.to_datetime(el, unit='s').strftime('%Y-%m-%d') for el in winner_timestamps]
    end_timestamp_formatted = pd.to_datetime(int(os.environ.get("END_TIMESTAMP")), unit='s').strftime('%Y-%m-%d')
    ticker_list = create_ticker_list()
    df = load_tickers()

    for ticker, amount, timestamp in zip(winner_tickers, winner_amounts, winner_timestamps_formatted):
        value_invested = df.loc[[timestamp], ticker_list[int(ticker)]][0]
        value_evaluation = df.loc[[end_timestamp_formatted], ticker_list[int(ticker)]][0]
        logging.info(f"Ticker: {ticker_list[int(ticker)]}, "
                     f"Amount: {amount}, "
                     f"Value at invested ({timestamp}): {value_invested}, "
                     f"Value at evaluation ({end_timestamp_formatted}): {value_evaluation}, "
                     f"Profit: {value_evaluation / value_invested * amount}")

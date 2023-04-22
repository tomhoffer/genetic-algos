import logging
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from typing import List, Hashable

import numpy as np
import pandas
import pandas as pd
from dotenv import load_dotenv
from numpy.random import default_rng

from src.generic.crossover import Crossover
from src.generic.executor import TrainingExecutor
from src.generic.model import Solution, Hyperparams
from src.generic.mutation import Mutation
from src.generic.selection import Selection

from src.tradingbot.config import Config
from src.tradingbot.decisions import TradingStrategies, Decision

logging.basicConfig(level=logging.INFO)
load_dotenv()


@dataclass
class Position:
    datetime: str
    amount: float


class TradingbotSolution(Solution):
    account_balance: float
    bought_positions: List[Position]

    def __init__(self, chromosome: np.ndarray, account_balance: float = 0):
        self.account_balance = account_balance
        self.bought_positions = []
        self.chromosome = chromosome

    def get_strategy_weights(self) -> np.ndarray:
        return self.chromosome

    def buy(self, datetime: str, amount: float):
        if self.account_balance < amount:
            return
        self.bought_positions.append(Position(datetime=datetime, amount=amount))
        self.account_balance -= amount
        logging.debug("Buying for %sUSD on date %s", amount, datetime)

    def sell(self, datetime: str):
        if len(self.bought_positions) == 0:
            return
        # Sell the first open buy position
        position_to_sell: Position = self.bought_positions[0]
        df: pandas.DataFrame = load_ticker_data()
        ticker_value_at_sell: float = df.at[datetime, f"{Config.get_value('TRADED_TICKER_NAME')}_Close"]
        ticker_value_at_buy: float = df.at[position_to_sell.datetime, f"{Config.get_value('TRADED_TICKER_NAME')}_Close"]
        profit: float = (ticker_value_at_sell / ticker_value_at_buy * position_to_sell.amount) - position_to_sell.amount

        self.bought_positions.pop(0)
        self.account_balance += profit
        logging.debug("Selling with profit of %s on date %s", profit, datetime)

    def sell_all(self, datetime: str):
        if len(self.bought_positions) == 0:
            return
        for _ in range(len(self.bought_positions)):
            self.sell(datetime)


@cache
def load_ticker_data(path: str = 'data.csv') -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['Date'], index_col=['Date'])

    # Remove columns with too few values
    # minimum_value_count = df.count().quantile(0.45)
    # return df.dropna(thresh=minimum_value_count, axis='columns')
    return df


def get_trading_strategy_method_names() -> List[str]:
    """
    :return: Names of all methods corresponding to trading strategies
    """
    return [method for method in dir(TradingStrategies) if method.startswith('decide')]


def timestamp_to_str(timestamp: int | pd.Timestamp | Hashable) -> str:
    if isinstance(timestamp, int):
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
    if isinstance(timestamp, pd.Timestamp):
        return timestamp.strftime('%Y-%m-%d')
    raise ValueError("Provided timestamp is not of type int or pd.Timestamp, unable to convert!")


def initial_population_generator() -> List[TradingbotSolution]:
    result: List[TradingbotSolution] = []
    num_of_strategies = len(get_trading_strategy_method_names())
    rng = np.random.default_rng()

    for i in range(Config.get_value('POPULATION_SIZE')):
        chromosome = rng.random(num_of_strategies)
        result.append(TradingbotSolution(chromosome))
        logging.debug("Generated random individual with chromosome %s", chromosome)
    return result


def fitness(chromosome: np.ndarray) -> float:
    end_date: str = timestamp_to_str(Config.get_value("END_TIMESTAMP"))
    start_date: str = timestamp_to_str(Config.get_value("START_TIMESTAMP"))
    evaluation_df: pd.DataFrame = load_ticker_data()[start_date:end_date]
    ts = TradingStrategies()
    solution = TradingbotSolution(chromosome=chromosome, account_balance=Config.get_value('BUDGET'))

    # Decide on each day of training period
    for row_datetime, row in evaluation_df.iterrows():
        # Make decision based on all trading strategies and their weights
        decisions: np.array = np.array(list(ts.perform_decisions_for_row(row).values()))
        decisions_sum = np.sum(np.multiply(chromosome, decisions))
        result = Decision.BUY if decisions_sum > 0 else Decision.SELL if decisions_sum < 0 else Decision.INCONCLUSIVE
        logging.debug("Result based on individual decisions: %s", result)

        if result == Decision.INCONCLUSIVE:
            continue
        elif result == Decision.BUY:
            solution.buy(datetime=timestamp_to_str(row_datetime), amount=Config.get_value("TRADE_SIZE"))
        elif result == Decision.SELL:
            solution.sell(datetime=timestamp_to_str(row_datetime))

    # Close all trades at the end of the period to evaluate solution performance
    solution.sell_all(datetime=end_date)
    solution.fitness = solution.account_balance
    return solution.fitness


def stopping_criteria_fn(solution: TradingbotSolution) -> bool:
    return True if solution.fitness > 500 else False


def chromosome_validator_fn(solution: TradingbotSolution) -> bool:
    return np.all((solution.chromosome >= 0) & (solution.chromosome <= 1))


if __name__ == "__main__":
    params = Hyperparams(crossover_fn=Crossover.uniform,
                         initial_population_generator_fn=initial_population_generator,
                         mutation_fn=Mutation.mutate_real_gaussian,
                         selection_fn=Selection.tournament,
                         fitness_fn=fitness, population_size=Config.get_value("POPULATION_SIZE"), elitism=5,
                         stopping_criteria_fn=stopping_criteria_fn, chromosome_validator_fn=chromosome_validator_fn)

    winner, success, id = TrainingExecutor.run((params, 1))
    logging.info("Found winner with weights %s and profit %s", winner.chromosome, winner.fitness)

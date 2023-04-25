import logging
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from typing import List, Hashable

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.generic.crossover import Crossover
from src.generic.executor import TrainingExecutor
from src.generic.hyperparams import HyperparamEvaluator
from src.generic.model import Solution, Hyperparams
from src.generic.mutation import Mutation
from src.generic.selection import Selection

from src.tradingbot.config import Config
from src.tradingbot.decisions import TradingStrategies, Decision

pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning
logging.basicConfig(level=logging.INFO)
load_dotenv()
backtesting = False


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
        df: pd.DataFrame = load_ticker_data()
        ticker_value_at_sell: float = df.at[datetime, f"{Config.get_value('TRADED_TICKER_NAME')}_Adj Close"]
        ticker_value_at_buy: float = df.at[
            position_to_sell.datetime, f"{Config.get_value('TRADED_TICKER_NAME')}_Adj Close"]
        profit: float = (ticker_value_at_sell / ticker_value_at_buy * position_to_sell.amount)

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
    return pd.read_csv(path, parse_dates=['Date'], index_col=['Date'])


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
        random_weights = rng.random(num_of_strategies)
        chromosome = random_weights / np.sum(random_weights)
        result.append(TradingbotSolution(chromosome))
        logging.debug("Generated random individual with chromosome %s", chromosome)
    return result


def fitness(chromosome: np.ndarray) -> float:
    if backtesting:
        end_date: str = timestamp_to_str(Config.get_value("BACKTEST_END_TIMESTAMP"))
        start_date: str = timestamp_to_str(Config.get_value("BACKTEST_START_TIMESTAMP"))
    else:
        end_date: str = timestamp_to_str(Config.get_value("END_TIMESTAMP"))
        start_date: str = timestamp_to_str(Config.get_value("START_TIMESTAMP"))

    evaluation_df: pd.DataFrame = load_ticker_data()[start_date:end_date]
    evaluation_df['datetime'] = evaluation_df.index
    row_np_index = dict(zip(evaluation_df.columns, list(range(0, len(evaluation_df.columns)))))
    evaluation_data: np.ndarray = evaluation_df.to_numpy()
    ts = TradingStrategies()
    solution = TradingbotSolution(chromosome=chromosome, account_balance=Config.get_value('BUDGET'))

    def decide_row(row: np.array):
        # Make decision based on all trading strategies and their weights
        decisions: np.array = np.array(list(ts.perform_decisions_for_row(row, row_np_index).values()))
        decisions_sum = np.sum(np.multiply(chromosome, decisions))
        result = Decision.BUY if decisions_sum > 0 else Decision.SELL if decisions_sum < 0 else Decision.INCONCLUSIVE
        logging.debug("Result based on individual decisions: %s", result)

        if result == Decision.INCONCLUSIVE:
            return
        elif result == Decision.BUY:
            solution.buy(datetime=timestamp_to_str(row[row_np_index['datetime']]),
                         amount=Config.get_value("TRADE_SIZE"))
        elif result == Decision.SELL:
            solution.sell(datetime=timestamp_to_str(row[row_np_index['datetime']]))

    np.apply_along_axis(decide_row, axis=1, arr=evaluation_data)
    # Close all trades at the end of the period to evaluate solution performance
    solution.sell_all(datetime=end_date)
    solution.fitness = solution.account_balance
    return solution.fitness


def stopping_criteria_fn(solution: TradingbotSolution) -> bool:
    return True if solution.fitness > 500 else False


def chromosome_validator_fn(solution: TradingbotSolution) -> bool:
    return np.all((solution.chromosome >= 0) & (solution.chromosome <= 1))


def mutate_gaussian(chromosome: np.ndarray) -> np.ndarray:
    return Mutation.mutate_real_gaussian(chromosome, use_abs=True, max=1.0, min=0)


def mutate_uniform(chromosome: np.ndarray) -> np.ndarray:
    return Mutation.mutate_real_uniform(chromosome, use_abs=True, max=1.0, min=0)


if __name__ == "__main__":
    params = Hyperparams(crossover_fn=Crossover.two_point,
                         initial_population_generator_fn=initial_population_generator,
                         mutation_fn=mutate,
                         selection_fn=Selection.tournament,
                         fitness_fn=fitness, population_size=Config.get_value("POPULATION_SIZE"), elitism=5,
                         stopping_criteria_fn=stopping_criteria_fn, chromosome_validator_fn=chromosome_validator_fn)

    logging.info("Training on period: %s - %s", timestamp_to_str(Config.get_value("START_TIMESTAMP")),
                 timestamp_to_str(Config.get_value("END_TIMESTAMP")))
    winner, success, id = TrainingExecutor.run((params, 1))
    logging.info("Found winner with weights %s and resulting account balance %s",
                 [el for el in zip(get_trading_strategy_method_names(), winner.chromosome)], winner.fitness)

    backtesting = True
    logging.info("Starting backtest...")
    logging.info("Resulting account balance over backtesting period: %s", fitness(winner.chromosome))
    """
    # selection_methods = [Selection.tournament, Selection.roulette, Selection.rank]
    selection_methods = [Selection.rank]
    # crossover_methods = [Crossover.two_point, Crossover.single_point, Crossover.uniform]
    crossover_methods = [Crossover.two_point]
    mutation_methods = [mutate]
    population_sizes = [10, 20]
    elitism_values = [1, 3]

    evaluator = HyperparamEvaluator(selection_methods=selection_methods, mutation_methods=mutation_methods,
                                    crossover_methods=crossover_methods, population_sizes=population_sizes,
                                    fitness_fn=fitness, initial_population_generation_fn=initial_population_generator,
                                    elitism_values=elitism_values, stopping_criteria_fn=stopping_criteria_fn,
                                    chromosome_validator_fn=chromosome_validator_fn)

    evaluator.grid_search_parallel()
    """

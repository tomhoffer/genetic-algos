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
from src.generic.model import Solution
from src.generic.mutation import Mutation
from src.generic.selection import Selection

from src.tradingbot.config import Config
from src.tradingbot.decisions import TradingStrategies, Decision
from src.tradingbot.exceptions import InvalidTradeActionException
from src.tradingbot.hyperparams import TradingBotHyperparamEvaluator, TradingBotHyperparams

pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning
logging.basicConfig(level=logging.INFO)
load_dotenv()
backtesting = False


@dataclass
class BuyPosition:
    datetime: str
    amount: float
    price_at_buy: float


class TradingbotSolution(Solution):
    account_balance: float
    bought_positions: List[BuyPosition]

    def __init__(self, chromosome: np.ndarray, account_balance: float = 0):
        self.account_balance = account_balance
        self.bought_positions = []
        self.chromosome = chromosome

    def get_strategy_weights(self) -> np.ndarray:
        return self.chromosome

    def buy(self, datetime: str, amount: float, price: float):

        if self.account_balance == 0:
            return

        amount_to_buy = amount
        if 0 < self.account_balance < amount:
            amount_to_buy = self.account_balance

        self.bought_positions.append(BuyPosition(datetime=datetime, amount=amount_to_buy, price_at_buy=price))
        self.account_balance -= amount_to_buy
        logging.debug("Buying for %sUSD on date %s with price %s", amount, datetime, price)

    def sell(self, datetime: str, index: int):
        """
        :param datetime: Datetime of sell action
        :param index: Index in the BuyPosition.bought_positions list
        """
        if len(self.bought_positions) == 0:
            return

        try:
            sold_position: BuyPosition = self.bought_positions.pop(index)
        except IndexError:
            raise InvalidTradeActionException(message=f"Attempting to sell a position on invalid index: {index}")

        df: pd.DataFrame = load_ticker_data()
        ticker_value_at_sell: float = df.at[datetime, f"{Config.get_value('TRADED_TICKER_NAME')}_Adj Close"]
        profit: float = (ticker_value_at_sell / sold_position.price_at_buy * sold_position.amount)
        self.account_balance += profit
        logging.debug("Selling with profit of %s on date %s", profit, datetime)

    def sell_all(self, datetime: str):
        if len(self.bought_positions) == 0:
            return
        for i in range(len(self.bought_positions)):
            self.sell(datetime=datetime, index=0)


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
        chromosome = rng.random(num_of_strategies)
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
        ticker_price: float = row[row_np_index[Config.get_value('TRADED_TICKER_NAME') + '_Adj Close']]
        # If stop_loss or take_profit criteria are met
        while True:
            sold = False
            for index, position in enumerate(solution.bought_positions):
                value_proportion: float = ticker_price / position.price_at_buy
                if value_proportion >= Config.get_value(
                        'TAKE_PROFIT_PROPORTION') or value_proportion <= Config.get_value('STOP_LOSS_PROPORTION'):
                    solution.sell(datetime=row[row_np_index['datetime']], index=index)
                    sold = True
                    break
            if not sold:
                break

        # Make decision based on all trading strategies and their weights
        decisions: np.array = np.array(list(ts.perform_decisions_for_row(row, row_np_index).values()))
        decisions_sum = np.sum(np.multiply(chromosome, decisions))
        decision_threshold: float = abs(Config.get_value('TRADE_ACTION_CONFIDENCE'))

        result = Decision.INCONCLUSIVE
        if decisions_sum > decision_threshold:
            result = Decision.BUY
        elif decisions_sum < (-1) * decision_threshold:
            result = Decision.SELL

        logging.debug("Result based on individual decisions: %s", result)

        if result == Decision.BUY:
            solution.buy(datetime=timestamp_to_str(row[row_np_index['datetime']]),
                         amount=Config.get_value("TRADE_SIZE"),
                         price=ticker_price)
        elif result == Decision.SELL:
            solution.sell(datetime=timestamp_to_str(row[row_np_index['datetime']]), index=0)

    np.apply_along_axis(decide_row, axis=1, arr=evaluation_data)
    # Close all trades at the end of the period to evaluate solution performance
    solution.sell_all(datetime=end_date)
    solution.fitness = solution.account_balance
    return solution.fitness


def stopping_criteria_fn(solution: TradingbotSolution) -> bool:
    return False
    # return True if solution.fitness > 2000 else False


def chromosome_validator_fn(solution: TradingbotSolution) -> bool:
    return np.all((solution.chromosome >= 0) & (solution.chromosome <= 1))


def mutate_gaussian(chromosome: np.ndarray) -> np.ndarray:
    return Mutation.mutate_real_gaussian(chromosome, use_abs=True, max=1.0, min=0)


def mutate_uniform(chromosome: np.ndarray) -> np.ndarray:
    return Mutation.mutate_real_uniform(chromosome, use_abs=True, max=1.0, min=0)


if __name__ == "__main__":
    params = TradingBotHyperparams(crossover_fn=Crossover.two_point,
                                   initial_population_generator_fn=initial_population_generator,
                                   mutation_fn=mutate_uniform,
                                   selection_fn=Selection.tournament,
                                   fitness_fn=fitness, population_size=Config.get_value("POPULATION_SIZE"), elitism=1,
                                   stopping_criteria_fn=stopping_criteria_fn,
                                   chromosome_validator_fn=chromosome_validator_fn,
                                   stop_loss_ratio=Config.get_value("STOP_LOSS_PROPORTION"),
                                   take_profit_ratio=Config.get_value("TAKE_PROFIT_PROPORTION"),
                                   trade_action_confidence=Config.get_value("TRADE_ACTION_CONFIDENCE"))

    logging.info("Training on period: %s - %s", timestamp_to_str(Config.get_value("START_TIMESTAMP")),
                 timestamp_to_str(Config.get_value("END_TIMESTAMP")))
    winner, success, id = TrainingExecutor.run((params, 1), return_global_winner=True)
    # winner, success, id = TrainingExecutor.run_parallel(params, return_global_winner=True)
    logging.info("Found winner with weights %s and resulting account balance %s",
                 [el for el in zip(get_trading_strategy_method_names(), winner.chromosome)], winner.fitness)

    backtesting = True
    logging.info("Starting backtest...")
    logging.info("Resulting account balance over backtesting period: %s", fitness(winner.chromosome))
    """
    # selection_methods = [Selection.tournament, Selection.roulette, Selection.rank]
    selection_methods = [Selection.rank]
    # crossover_methods = [Crossover.two_point, Crossover.single_point, Crossover.uniform]
    crossover_methods = [Crossover.single_point]
    mutation_methods = [mutate_uniform]
    population_sizes = [10]
    elitism_values = [1]
    stop_loss_ratios = [0.5, 0.6, 0.7, 0.8, 0.9]
    take_profit_ratios = [1.1, 1.2, 1.3, 1.5, 1.7, 2.0]
    trade_action_confidences = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1]

    evaluator = TradingBotHyperparamEvaluator(selection_method=selection_methods, mutation_method=mutation_methods,
                                              crossover_method=crossover_methods, population_size=population_sizes,
                                              fitness_fn=fitness,
                                              initial_population_generation_fn=initial_population_generator,
                                              elitism_value=elitism_values, stopping_criteria_fn=stopping_criteria_fn,
                                              chromosome_validator_fn=chromosome_validator_fn,
                                              stop_loss_ratio=stop_loss_ratios, take_profit_ratio=take_profit_ratios,
                                              trade_action_confidence=trade_action_confidences)

    evaluator.grid_search_parallel()
    """

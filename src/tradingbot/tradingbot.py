import logging
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from typing import List, Hashable, Dict, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import matplotlib.pyplot as plt

from src.generic.crossover import Crossover
from src.generic.executor import TrainingExecutor
from src.generic.model import Solution
from src.generic.mutation import Mutation
from src.generic.selection import Selection

from src.tradingbot.config import Config
from src.tradingbot.decisions import TradingStrategies, Decision
from src.tradingbot.decorators import np_cache
from src.tradingbot.exceptions import InvalidTradeActionException
from src.tradingbot.hyperparams import TradingBotHyperparamEvaluator, TradingBotHyperparams

pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning
logging.basicConfig(level=logging.INFO)
load_dotenv()


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

    def parse_chromosome_trade_size(self):
        return self.chromosome[-1]

    def parse_chromosome_stop_loss(self):
        return self.chromosome[-2]

    def parse_chromosome_take_profit(self):
        return self.chromosome[-3]

    def parse_strategy_weights(self):
        return self.chromosome[:-3]


@cache
def load_ticker_data(
        path: str = f"data/data-{Config.get_value('TRADED_TICKER_NAME')}.csv") -> pd.DataFrame:
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
        strategy_weights: np.ndarray = rng.random(num_of_strategies)
        take_profit_ratio: float = rng.uniform(low=1, high=2)
        stop_loss_ratio: float = rng.uniform(low=0, high=1)
        trade_size: float = rng.uniform(low=10, high=100)
        chromosome: np.ndarray = np.append(strategy_weights, [take_profit_ratio, stop_loss_ratio, trade_size])
        result.append(TradingbotSolution(chromosome))
        logging.debug("Generated random individual with chromosome %s", chromosome)
    return result


@np_cache
def fitness(chromosome: np.ndarray, backtesting: bool = False) -> float | Tuple[float, List]:
    transaction_log: List[Dict] = []
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
    take_profit_proportion: float = solution.parse_chromosome_take_profit()
    stop_loss_proportion: float = solution.parse_chromosome_stop_loss()
    trade_size: float = solution.parse_chromosome_trade_size()

    def decide_row(row: np.array):
        ticker_price: float = row[row_np_index[Config.get_value('TRADED_TICKER_NAME') + '_Adj Close']]
        row_datetime: str = timestamp_to_str(row[row_np_index['datetime']])

        # If stop_loss or take_profit criteria are met
        if solution.bought_positions:
            while True:
                sold = False
                for index, position in enumerate(solution.bought_positions):
                    value_proportion: float = ticker_price / position.price_at_buy
                    if value_proportion >= take_profit_proportion or value_proportion <= stop_loss_proportion:
                        solution.sell(datetime=row_datetime, index=index)
                        sold = True
                        break
                if not sold:
                    break

        # Make decision based on all trading strategies and their weights
        decisions: np.array = np.array(list(ts.perform_decisions_for_row(row, row_np_index).values()))
        decisions_sum = np.sum(np.multiply(solution.parse_strategy_weights(), decisions))

        result = Decision.INCONCLUSIVE
        if decisions_sum > 0:
            # Buy based on confidence
            trade_size_adjusted = trade_size * (1 + decisions_sum)
            solution.buy(datetime=row_datetime, amount=trade_size_adjusted, price=ticker_price)
            result = Decision.BUY
        elif decisions_sum < 0:
            # Sell oldest trade
            solution.sell(datetime=row_datetime, index=0)
            result = Decision.SELL

        if backtesting:
            transaction_log.append({'type': result, 'datetime': row_datetime, 'price': ticker_price})

    np.apply_along_axis(decide_row, axis=1, arr=evaluation_data)
    # Close all trades at the end of the period to evaluate solution performance
    solution.sell_all(datetime=end_date)
    solution.fitness = solution.account_balance

    if backtesting:
        return solution.fitness, transaction_log

    return solution.fitness


def stopping_criteria_fn(solution: TradingbotSolution) -> bool:
    return True if solution.fitness > 100000 else False


def chromosome_validator_fn(solution: TradingbotSolution) -> bool:
    strategy_weights: np.ndarray = solution.parse_strategy_weights()
    take_profit_ratio: float = solution.parse_chromosome_take_profit()
    stop_loss_ratio: float = solution.parse_chromosome_stop_loss()
    trade_size: float = solution.parse_chromosome_trade_size()
    return np.all((strategy_weights >= 0) & (strategy_weights <= 1)) and \
        (1 <= take_profit_ratio < 2) and \
        (0 <= stop_loss_ratio < 1) and (0 < trade_size)


def mutate_gaussian(chromosome: np.ndarray) -> np.ndarray:
    return Mutation.mutate_real_gaussian(chromosome, use_abs=True, max=1.0, min=0)  # TODO adjust


def mutate_uniform(chromosome: np.ndarray) -> np.ndarray:
    solution = TradingbotSolution(chromosome=chromosome)
    strategy_weights: np.ndarray = solution.parse_strategy_weights()
    take_profit_ratio: float = solution.parse_chromosome_take_profit()
    stop_loss_ratio: float = solution.parse_chromosome_take_profit()
    trade_size: float = solution.parse_chromosome_trade_size()

    mutated_weights: np.ndarray = Mutation.mutate_real_uniform(strategy_weights, use_abs=True, max=1.0, min=0)
    mutated_stop_loss: np.ndarray = Mutation.mutate_real_uniform(np.asarray([stop_loss_ratio]), use_abs=True, max=1.0,
                                                                 min=0)
    mutated_take_profit: np.ndarray = Mutation.mutate_real_uniform(np.asarray([take_profit_ratio]), use_abs=True, max=2,
                                                                   min=1)
    mutated_trade_size: np.ndarray = Mutation.mutate_real_uniform(np.asarray([trade_size]), use_abs=True, max=100,
                                                                  min=1)
    return np.concatenate((mutated_weights, mutated_take_profit, mutated_stop_loss, mutated_trade_size))


def backtest(winner: Solution):
    print("Starting backtest...")
    winner_fitness, transaction_log = fitness(winner.chromosome, backtesting=True)
    print(f"Resulting account balance over backtesting period: {winner_fitness}")

    fig, ax = plt.subplots()
    end_date: str = timestamp_to_str(Config.get_value("BACKTEST_END_TIMESTAMP"))
    start_date: str = timestamp_to_str(Config.get_value("BACKTEST_START_TIMESTAMP"))
    ticker_df: pd.DataFrame = load_ticker_data()[start_date:end_date]
    ax.plot(ticker_df.index, ticker_df[f"{Config.get_value('TRADED_TICKER_NAME')}_Adj Close"])

    buys = [(datetime.strptime(log['datetime'], '%Y-%m-%d'), log['price']) for log in transaction_log if
            log['type'] == Decision.BUY]
    sells = [(datetime.strptime(log['datetime'], '%Y-%m-%d'), log['price']) for log in transaction_log if
             log['type'] == Decision.SELL]

    for buy, sell in zip(buys, sells):
        plt.scatter(x=buy[0], y=buy[1], color='g')
        plt.scatter(x=sell[0], y=sell[1], color='r')
    plt.show()


if __name__ == "__main__":
    params = TradingBotHyperparams(crossover_fn=Crossover.two_point,
                                   initial_population_generator_fn=initial_population_generator,
                                   mutation_fn=mutate_uniform,
                                   selection_fn=Selection.tournament,
                                   fitness_fn=fitness, population_size=Config.get_value("POPULATION_SIZE"), elitism=1,
                                   stopping_criteria_fn=stopping_criteria_fn,
                                   chromosome_validator_fn=chromosome_validator_fn)

    print(
        f"Training on period: {timestamp_to_str(Config.get_value('START_TIMESTAMP'))} - {timestamp_to_str(Config.get_value('END_TIMESTAMP'))}")
    winner, success, id = TrainingExecutor.run((params, 1), return_global_winner=True)
    # winner, success, id = TrainingExecutor.run_parallel(params, return_global_winner=True)
    print(
        f"Found winner with weights {[el for el in zip(get_trading_strategy_method_names(), winner.chromosome)]} and resulting account balance {winner.fitness}")

    backtest(winner)

    """
    # selection_methods = [Selection.tournament, Selection.roulette, Selection.rank]
    selection_methods = [Selection.rank]
    # crossover_methods = [Crossover.two_point, Crossover.single_point, Crossover.uniform]
    crossover_methods = [Crossover.two_point]
    mutation_methods = [mutate_uniform]
    population_sizes = [500]
    elitism_values = [5]

    evaluator = TradingBotHyperparamEvaluator(selection_method=selection_methods, mutation_method=mutation_methods,
                                              crossover_method=crossover_methods, population_size=population_sizes,
                                              fitness_fn=fitness,
                                              initial_population_generation_fn=initial_population_generator,
                                              elitism_value=elitism_values, stopping_criteria_fn=stopping_criteria_fn,
                                              chromosome_validator_fn=chromosome_validator_fn)

    evaluator.grid_search_parallel(return_global_winner=True)
    """

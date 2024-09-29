import logging
import random
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from functools import cache
from typing import List, Hashable, Dict
from dotenv import load_dotenv
from src.generic.model import Solution
from src.generic.mutation import Mutation
from src.tradingbot.enums import SellTrigger
from src.tradingbot.trade_logger import TradeLogger
from src.tradingbot.config import Config
from src.tradingbot.decisions import TradingStrategies
from src.tradingbot.exceptions import InvalidTradeActionException
from src.tradingbot.repository import TradingdataRepository

pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning
trading_data_repository = TradingdataRepository()
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

        amount_to_buy = amount
        if not Config.get_value("ALLOW_BUDGET_EXCEED"):
            if self.account_balance == 0:
                return

            if 0 < self.account_balance < amount:
                amount_to_buy = self.account_balance

        self.bought_positions.append(BuyPosition(datetime=datetime, amount=amount_to_buy, price_at_buy=price))
        self.account_balance -= amount_to_buy
        trade_logger = TradeLogger()
        trade_logger.log_buy_position(datetime=datetime, price=price)
        logging.debug("Buying for %sUSD on date %s with price %s", amount, datetime, price)

    def sell(self, datetime: str, index: int, trigger: SellTrigger = SellTrigger.SHORT) -> float:
        """
        :param datetime: Datetime of sell action
        :param index: Index in the BuyPosition.bought_positions list
        :return: Profit of the sell action
        """
        if len(self.bought_positions) == 0:
            return

        try:
            sold_position: BuyPosition = self.bought_positions.pop(index)
        except IndexError:
            raise InvalidTradeActionException(message=f"Attempting to sell a position on invalid index: {index}")

        df: pd.DataFrame = trading_data_repository.load_ticker_data()
        ticker_value_at_sell: float = df.at[datetime, f"{Config.get_value('TRADED_TICKER_NAME')}_Adj Close"]
        value_at_sell: float = (ticker_value_at_sell / sold_position.price_at_buy * sold_position.amount)
        self.account_balance += value_at_sell
        logging.debug("Selling with profit of %s on date %s", value_at_sell, datetime)
        profit = (ticker_value_at_sell * sold_position.amount) - (sold_position.price_at_buy * sold_position.amount)
        l = TradeLogger()
        l.log_sell_position(datetime=datetime, price=value_at_sell, profit=profit, trigger=trigger)
        return profit

    def sell_all(self, datetime: str):
        if len(self.bought_positions) == 0:
            return
        for i in range(len(self.bought_positions)):
            self.sell(datetime=datetime, index=0)

    def sell_position_with_highest_profit(self, datetime: str) -> float:
        """
        :param datetime: Datetime of sell action
        :return: Profit of the sell action
        """
        if len(self.bought_positions) == 0:
            return
        cheapest_buy_position: BuyPosition = min(self.bought_positions, key=lambda x: x.price_at_buy)
        profit: float = self.sell(datetime=datetime, index=self.bought_positions.index(cheapest_buy_position),
                                  trigger=SellTrigger.SHORT)
        return profit

    def parse_chromosome_trade_size(self) -> float:
        return self.chromosome[-1]

    def parse_chromosome_stop_loss(self) -> float:
        return self.chromosome[-2]

    def parse_chromosome_take_profit(self) -> float:
        return self.chromosome[-3]

    def parse_strategy_weights(self) -> np.ndarray:
        return self.chromosome[:-3]

    def serialize_to_file(self, path: str):
        trading_strategy_names: List[str] = get_trading_strategy_method_names()
        trading_strategy_weights: np.ndarray = self.parse_strategy_weights().reshape(1, -1)  # Reshape to 2D with 1 row
        df = pd.DataFrame(trading_strategy_weights, columns=trading_strategy_names)
        df.to_csv(path, index=False)


def get_trading_strategy_method_names() -> List[str]:
    """
    :return: Names of all methods corresponding to trading strategies
    """
    return [method for method in dir(TradingStrategies) if method.startswith('decide')]


@cache
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
        stop_loss_ratio: float = rng.uniform(low=0, high=0.99)
        trade_size: float = rng.uniform(low=10, high=100)
        chromosome: np.ndarray = np.append(strategy_weights, [take_profit_ratio, stop_loss_ratio, trade_size])
        result.append(TradingbotSolution(np.around(chromosome, decimals=2)))
        logging.debug("Generated random individual with chromosome %s", chromosome)
    return result


def decide_row(row: np.array, row_np_index: Dict, solution: TradingbotSolution):
    ticker_price: float = row[row_np_index[Config.get_value('TRADED_TICKER_NAME') + '_Adj Close']]
    row_datetime: str = timestamp_to_str(row[row_np_index['datetime']])
    take_profit_proportion: float = solution.parse_chromosome_take_profit()
    stop_loss_proportion: float = solution.parse_chromosome_stop_loss()
    trade_size: float = solution.parse_chromosome_trade_size()
    ts = TradingStrategies()
    trade_logger = TradeLogger()

    # If stop_loss or take_profit criteria are met
    if solution.bought_positions:
        while True:
            sold = False
            for index, position in enumerate(solution.bought_positions):
                value_proportion: float = ticker_price / position.price_at_buy
                if value_proportion >= take_profit_proportion:
                    solution.sell(datetime=row_datetime, index=index, trigger=SellTrigger.TAKE_PROFIT)
                    sold = True
                    break
                elif value_proportion <= stop_loss_proportion:
                    solution.sell(datetime=row_datetime, index=index, trigger=SellTrigger.STOP_LOSS)
                    sold = True
                    break
            if not sold:
                break

    # Make decision based on all trading strategies and their weights
    decisions: np.array = np.array(list(ts.perform_decisions_for_row(row, row_np_index).values()))
    decisions_sum = np.sum(np.multiply(solution.parse_strategy_weights(), decisions))

    result = Decision.INCONCLUSIVE
    profit = -np.inf
    if decisions_sum > 0:
        # Buy based on confidence
        trade_size_adjusted = trade_size * (1 + decisions_sum)
        solution.buy(datetime=row_datetime, amount=trade_size_adjusted, price=ticker_price)
        result = Decision.BUY
    elif decisions_sum < 0:
        # Sell position with the highest profit
        # solution.sell(datetime=row_datetime, index=0)
        profit: float = solution.sell_position_with_highest_profit(datetime=row_datetime)
        result = Decision.SELL



def perform_fitness(start_date: str, end_date: str, chromosome: np.ndarray) -> float:
    evaluation_df: pd.DataFrame = trading_data_repository.load_ticker_data(start_date=start_date, end_date=end_date)
    evaluation_df['datetime'] = evaluation_df.index
    row_np_index = dict(zip(evaluation_df.columns, list(range(0, len(evaluation_df.columns)))))
    evaluation_data: np.ndarray = evaluation_df.to_numpy()
    solution = TradingbotSolution(chromosome=chromosome, account_balance=Config.get_value('BUDGET'))

    np.apply_along_axis(decide_row, axis=1, arr=evaluation_data, row_np_index=row_np_index, solution=solution)
    # Close all trades at the end of the period to evaluate solution performance
    solution.sell_all(datetime=end_date)

    solution.fitness = solution.account_balance
    return solution.fitness


def fitness(chromosome: np.ndarray) -> float:
    end_date: str = timestamp_to_str(Config.get_value("END_TIMESTAMP"))
    start_date: str = timestamp_to_str(Config.get_value("START_TIMESTAMP"))
    return perform_fitness(start_date=start_date, end_date=end_date, chromosome=chromosome)


def stopping_criteria_fn(solution: TradingbotSolution) -> bool:
    return False
    # return True if solution.fitness > 2000 else False


def chromosome_validator_fn(solution: TradingbotSolution) -> bool:
    strategy_weights: np.ndarray = solution.parse_strategy_weights()
    take_profit_ratio: float = solution.parse_chromosome_take_profit()
    stop_loss_ratio: float = solution.parse_chromosome_stop_loss()
    trade_size: float = solution.parse_chromosome_trade_size()
    return np.all((strategy_weights >= 0) & (strategy_weights <= 1)) and \
        (1 <= take_profit_ratio <= 2) and \
        (0 <= stop_loss_ratio < 1) and (0 < trade_size)


def mutate_gaussian(chromosome: np.ndarray) -> np.ndarray:
    return np.around(Mutation.mutate_real_gaussian(chromosome, use_abs=True, max=1.0, min=0), decimals=2)  # TODO adjust


def mutate_uniform(chromosome: np.ndarray) -> np.ndarray:
    if not random.random() < Config.get_value("P_MUTATION"):
        return chromosome

    result = chromosome
    solution = TradingbotSolution(chromosome=chromosome)
    strategy_weights: np.ndarray = solution.parse_strategy_weights()
    take_profit_ratio: float = solution.parse_chromosome_take_profit()
    stop_loss_ratio: float = solution.parse_chromosome_take_profit()
    trade_size: float = solution.parse_chromosome_trade_size()
    part_to_mutate: int = random.randint(1, 4)

    if part_to_mutate == 1:
        mutated_weights = Mutation.mutate_real_uniform(strategy_weights, use_abs=True, max=1.0, min=0, force=True)
        result = np.concatenate((mutated_weights, [take_profit_ratio], [stop_loss_ratio], [trade_size]))

    elif part_to_mutate == 2:
        mutated_take_profit_ratio = Mutation.mutate_real_uniform(np.asarray([take_profit_ratio]), use_abs=True,
                                                                 max=1.0, min=0, force=True)
        result = np.concatenate((strategy_weights, mutated_take_profit_ratio, [stop_loss_ratio], [trade_size]))

    elif part_to_mutate == 3:

        mutated_stop_loss_ratio = Mutation.mutate_real_uniform(np.asarray([stop_loss_ratio]), use_abs=True, max=1.0,
                                                               min=0, force=True)
        result = np.concatenate((strategy_weights, [take_profit_ratio], mutated_stop_loss_ratio, [trade_size]))

    elif part_to_mutate == 4:
        mutated_trade_size = Mutation.mutate_real_uniform(np.asarray([trade_size]), use_abs=True, max=1.0, min=0,
                                                          force=True)
        result = np.concatenate((strategy_weights, [take_profit_ratio], [stop_loss_ratio], mutated_trade_size))

    return np.around(result, decimals=2)

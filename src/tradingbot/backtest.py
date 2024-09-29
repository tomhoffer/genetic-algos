from datetime import datetime
from typing import List

import numpy as np
import termplotlib as tpl
import pandas as pd
from matplotlib import pyplot as plt
from numpy import inf

from src.tradingbot.config import Config
from src.tradingbot.repository import TradingdataRepository
from src.tradingbot.trade_logger import TradeLogger
from src.tradingbot.tradingbot import TradingbotSolution, timestamp_to_str, perform_fitness


class BacktestExecutor:
    trading_data_repository: TradingdataRepository
    trade_logger: TradeLogger

    def __init__(self):
        self.trading_data_repository = TradingdataRepository()
        self.trade_logger = TradeLogger()

    def compute_win_rate(self) -> float:
        try:
            return len(self.trade_logger.get_all_profitable_sells()) / len(self.trade_logger.get_all_sells())
        except ZeroDivisionError:
            return -inf

    def compute_payoff_ratio(self) -> float:
        try:
            return self.trade_logger.get_avg_profit_per_sell() / self.trade_logger.get_avg_loss_per_sell()
        except ZeroDivisionError:
            return -inf

    def compute_profit_ratio(self) -> float:
        try:
            return sum([el['profit'] for el in self.trade_logger.get_all_profitable_sells()]) / sum(
                [el['profit'] for el in self.trade_logger.get_all_loosing_sells()])
        except ZeroDivisionError:
            return -inf

    def get_largest_profitable_trade(self) -> float:
        return max([el['profit'] for el in self.trade_logger.get_all_profitable_sells()])

    def get_largest_loosing_trade(self) -> float:
        return min([el['profit'] for el in self.trade_logger.get_all_loosing_sells()])

    def _plot_terminal(self, x: np.ndarray | List[float], y: np.ndarray | List[float], label="data", width=50,
                       height=15):
        fig = tpl.figure()
        fig.plot(x=x, y=y, label=label, width=width, height=height)
        fig.show()

    def backtest(self, winner: TradingbotSolution, plot=False, print_results=False,
                 start_date: str = timestamp_to_str(Config.get_value("BACKTEST_START_TIMESTAMP")),
                 end_date: str = timestamp_to_str(Config.get_value("BACKTEST_END_TIMESTAMP"))) -> float:

        logger = TradeLogger()
        logger.reset()
        logger.enable()
        winner_fitness = perform_fitness(start_date=start_date, end_date=end_date, chromosome=winner.chromosome)
        logger.disable()

        # Plot the graph with buys and sells
        if plot:
            buys = self.trade_logger.get_transaction_buys()
            sells = self.trade_logger.get_all_sells()

            fig, ax = plt.subplots()
            ticker_df: pd.DataFrame = self.trading_data_repository.load_ticker_data(start_date=start_date,
                                                                                    end_date=end_date)
            ax.plot(ticker_df.index, ticker_df[f"{Config.get_value('TRADED_TICKER_NAME')}_Adj Close"])

            for buy, sell in zip(buys, sells):
                plt.scatter(x=datetime.strptime(buy['datetime'], format="%Y-%m-%d"), y=buy['price'],
                            color='g')  # TODO fix
                plt.scatter(x=datetime.strftime(buy['datetime'], format="%Y-%m-%d"), y=sell['price'], color='r')
            plt.title(f"Resulting balance: {winner_fitness}")
            plt.show()
            plt.savefig("backtest.png")

        if print_results:
            print("------------------------------------------------------------")
            print(f"Backtest statistics for period {start_date} - {end_date}")
            print("Buys: ", len(self.trade_logger.get_transaction_buys()))
            print("Sells: ", len(self.trade_logger.get_transaction_sells()))
            print("Loosing: ", len(self.trade_logger.get_all_loosing_sells()))
            print("Stop loss hits: ", len(self.trade_logger.get_stop_loss_sells()))
            print("Take profit hits: ", len(self.trade_logger.get_take_profit_sells()))
            print("Account status: ", winner_fitness)
            print("Win rate: ", self.compute_win_rate())
            print("Payoff rate (> 1): ", self.compute_payoff_ratio())
            print("Profit ratio (> 1.75): ", self.compute_profit_ratio())
            print("Largest profitable trade: ", self.get_largest_profitable_trade())
            print("Largest loosing trade: ", self.get_largest_loosing_trade())
            self._plot_terminal(x=np.arange(len(self.trade_logger.account_status_history)),
                                y=self.trade_logger.account_status_history, label="Account history")

        return winner_fitness

from datetime import datetime
from typing import List, Dict, Tuple

import pandas as pd
from matplotlib import pyplot as plt

from src.tradingbot.config import Config
from src.tradingbot.enums import Decision
from src.tradingbot.repository import TradingdataRepository
from src.tradingbot.tradingbot import TradingbotSolution, timestamp_to_str, perform_fitness


class BacktestExecutor:
    transaction_log: List[Dict]
    trading_data_repository: TradingdataRepository

    def __init__(self):
        self.trading_data_repository = TradingdataRepository()
        self.transaction_log = []

    def _calculate_win_ratio(self):
        pass

    def _get_transaction_stop_loss_sells(self) -> List[Tuple[datetime, float, float]]:
        return [(datetime.strptime(log['datetime'], '%Y-%m-%d'), log['price'], log['profit']) for log in
                self.transaction_log if
                'event' in log and log['event'] == 'trigger' and log['profit'] < 0]

    def _get_transaction_take_profit_sells(self) -> List[Tuple[datetime, float, float]]:
        return [(datetime.strptime(log['datetime'], '%Y-%m-%d'), log['price'], log['profit']) for log in
                self.transaction_log if
                'event' in log and log['event'] == 'trigger' and log['profit'] > 0]

    def _get_transaction_sells(self) -> List[Tuple[datetime, float, float]]:
        return [(datetime.strptime(log['datetime'], '%Y-%m-%d'), log['price'], log['profit']) for log in
                self.transaction_log if log['type'] == Decision.SELL]

    def _get_transaction_buys(self) -> List[Tuple[datetime, float]]:
        return [(datetime.strptime(log['datetime'], '%Y-%m-%d'), log['price']) for log in
                self.transaction_log if log['type'] == Decision.BUY]

    def _get_profitable_sells(self) -> List[Tuple[datetime, float, float]]:

        return [(datetime.strptime(log['datetime'], '%Y-%m-%d'), log['price'], log['profit']) for log in
                self.transaction_log if
                log['type'] == Decision.SELL and log['profit'] is not None and log['profit'] > 0]

    def _get_loosing_sells(self) -> List[Tuple[datetime, float, float]]:

        return [(datetime.strptime(log['datetime'], '%Y-%m-%d'), log['price'], log['profit']) for log in
                self.transaction_log if
                log['type'] == Decision.SELL and log['profit'] is not None and log['profit'] < 0]

    def compute_win_rate(self) -> float:
        return len(self._get_profitable_sells()) / len(self.transaction_log)

    def compute_payoff_ratio(self) -> float:
        profitable_sells = self._get_profitable_sells()
        avg_profit_per_trade: float = sum([el[2] for el in profitable_sells]) / len(profitable_sells)
        loosing_sells = self._get_loosing_sells()
        avg_loss_per_trade: float = sum([abs(el[2]) for el in loosing_sells]) / len(loosing_sells)
        return avg_profit_per_trade / avg_loss_per_trade

    def get_largest_profitable_trade(self) -> float:
        return max([el[2] for el in (self._get_profitable_sells() + self._get_transaction_take_profit_sells())])

    def get_largest_loosing_trade(self) -> float:
        return min([el[2] for el in (self._get_loosing_sells() + self._get_transaction_stop_loss_sells())])

    def backtest(self, winner: TradingbotSolution, plot=False,
                 start_date: str = timestamp_to_str(Config.get_value("BACKTEST_START_TIMESTAMP")),
                 end_date: str = timestamp_to_str(Config.get_value("BACKTEST_END_TIMESTAMP"))) -> float:

        winner_fitness = perform_fitness(start_date=start_date, end_date=end_date, chromosome=winner.chromosome,
                                         transaction_log=self.transaction_log)

        buys = self._get_transaction_buys()
        sells = self._get_transaction_sells()

        # Plot the graph with buys and sells
        if plot:
            fig, ax = plt.subplots()
            ticker_df: pd.DataFrame = self.trading_data_repository.load_ticker_data(start_date=start_date,
                                                                                    end_date=end_date)
            ax.plot(ticker_df.index, ticker_df[f"{Config.get_value('TRADED_TICKER_NAME')}_Adj Close"])

            for buy, sell in zip(buys, sells):
                plt.scatter(x=buy[0], y=buy[1], color='g')
                plt.scatter(x=sell[0], y=sell[1], color='r')
            plt.title(f"Resulting balance: {winner_fitness}")
            plt.show()
            plt.savefig("backtest.png")

        print("------------------------------------------------------------")
        print(f"Backtest statistics for period {start_date} - {end_date}")
        print("Buys: ", len(self._get_transaction_buys()))
        print("Sells: ", len(self._get_transaction_sells()))
        print("Stop loss hits: ", len(self._get_transaction_stop_loss_sells()))
        print("Take profit hits: ", len(self._get_transaction_take_profit_sells()))
        print("Account status: ", winner_fitness)
        print("Win rate: ", self.compute_win_rate())
        print("Payoff rate (> 1): ", self.compute_payoff_ratio())
        print("Largest profitable trade: ", self.get_largest_profitable_trade())
        print("Largest loosing trade: ", self.get_largest_loosing_trade())

        return winner_fitness

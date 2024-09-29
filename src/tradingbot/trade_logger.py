from statistics import mean
from typing import List, Dict

from src.tradingbot.enums import Decision, SellTrigger


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class TradeLogger(metaclass=SingletonMeta):
    trading_log: List[Dict] = []
    account_status_history: List[float] = []
    enabled = False

    def log_buy_position(self, datetime: str, price: float, account_status_after_transaction: float):
        if self.enabled:
            self.trading_log.append({"datetime": datetime, "price": price, "type": Decision.BUY})

    def log_sell_position(self, datetime: str, price: float, profit: float, account_status_after_transaction: float,
                          trigger: SellTrigger = SellTrigger.SHORT):
        if self.enabled:
            self.trading_log.append(
                {"datetime": datetime, "price": price, "type": Decision.SELL, "trigger": trigger, "profit": profit})

    def log_account_status(self, balance: float):
        if self.enabled:
            self.account_status_history.append(balance)

    def get_log_count(self):
        return len(self.trading_log)

    def get_all_sells(self) -> List[Dict]:
        return [log for log in self.trading_log if log['type'] == Decision.SELL]

    def get_transaction_sells(self) -> List[Dict]:
        return [log for log in self.get_all_sells() if log['trigger'] == SellTrigger.SHORT]

    def get_transaction_buys(self) -> List[Dict]:
        return [log for log in self.trading_log if log['type'] == Decision.BUY]

    def get_stop_loss_sells(self) -> List[Dict]:
        return [log for log in self.get_all_sells() if log['trigger'] == SellTrigger.STOP_LOSS]

    def get_take_profit_sells(self) -> List[Dict]:
        return [log for log in self.get_all_sells() if log['trigger'] == SellTrigger.TAKE_PROFIT]

    def get_all_profitable_sells(self) -> List[Dict]:
        return [log for log in self.get_all_sells() if log['profit'] > 0]

    def get_all_loosing_sells(self) -> List[Dict]:
        return [log for log in self.get_all_sells() if log['profit'] < 0]

    def get_avg_profit_per_sell(self) -> float:
        return mean(el['profit'] for el in self.get_all_profitable_sells())

    def get_avg_loss_per_sell(self) -> float:
        return mean(el['profit'] for el in self.get_all_loosing_sells())

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def reset(self):
        self.trading_log = []
        self.account_status_history = []

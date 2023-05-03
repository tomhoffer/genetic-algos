import os
from functools import cache
from typing import get_type_hints, Any
from dotenv import load_dotenv

from src.generic.helpers import eval_bool

load_dotenv()


class AppConfigError(Exception):
    pass


# AppConfig class with required fields, default values, type checking, and typecasting for int and bool values
class Config:
    BUDGET: float
    TRADE_SIZE: float
    START_TIMESTAMP: int
    END_TIMESTAMP: int
    BACKTEST_END_TIMESTAMP: int
    BACKTEST_START_TIMESTAMP: int
    MAX_ITERS: int
    N_PROCESSES: int
    POPULATION_SIZE: int
    ENABLE_WANDB = False
    ELITISM = False
    P_MUTATION: float
    TRADED_TICKER_NAME: str

    """
    Map environment variables to class fields according to these rules:
      - Field won't be parsed unless it has a type annotation
      - Class field and environment variable name are the same
    """

    @staticmethod
    @lru_cache(maxsize=0 if "pytest" in sys.modules else 256)
    def get_value(value: str) -> Any:
        # Cast env var value to expected type and raise AppConfigError on failure
        var_type = get_type_hints(Config)[value]
        try:
            if var_type == bool:
                value = eval_bool(os.environ.get(value))
            else:
                value = var_type(os.environ.get(value))
            return value

        except ValueError:
            raise AppConfigError('Unable to cast value of to type "{}" for "{}" field'.format(
                var_type,
                value)
            )

    def __repr__(self):
        return str(self.__dict__)

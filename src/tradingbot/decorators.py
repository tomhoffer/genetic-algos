import sys
from functools import lru_cache, wraps
from typing import Callable
import numpy as np

from src.tradingbot import redis_connector
from src.generic.helpers import hash_chromosome

redis_conn = redis_connector.connect()


def np_cache(function, size=0 if "pytest" in sys.modules else 3500000):
    @lru_cache(maxsize=size)
    def cached_wrapper(hashable_array, *args, **kwargs):
        array = np.array(hashable_array)
        return function(array, *args, **kwargs)

    @wraps(function)
    def wrapper(array, *args, **kwargs):
        return cached_wrapper(tuple(array), *args, **kwargs)

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


def redis_cache(function: Callable):
    """
    This decorator caches results of a function receiving numpy array as an argument into Redis
    """

    @wraps(function)  # TODO close connections
    def wrapper(*args, **kwargs):
        args_hash: str = hash_chromosome(args[0]) + f"-{str(kwargs)}"
        cached_result = redis_conn.get(args_hash)

        if cached_result is None:
            # Run the function and cache the result for next time.
            value = function(*args, **kwargs)
            redis_conn.set(args_hash, str(value))
        else:
            # Skip the function entirely and use the cached value instead.
            value = float(cached_result)
        return value

    return wrapper

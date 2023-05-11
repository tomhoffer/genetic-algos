import sys
from functools import lru_cache, wraps

import numpy as np


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

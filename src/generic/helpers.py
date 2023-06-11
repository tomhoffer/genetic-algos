from hashlib import sha1

import numpy as np


def eval_bool(value: str) -> bool:
    """
    Function used to parse boolean values in env variables
    :param value: string value of the env variable
    :return: boolean interpretation of the value
    """
    try:
        if value.upper() == "TRUE":
            return True
        else:
            return False
    except AttributeError:
        # Empty variable
        return False


def hash_chromosome(chromosome: np.ndarray) -> str:
    return sha1(chromosome).hexdigest()

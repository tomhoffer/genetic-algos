import logging
import random

import numpy as np

from src.decorators import validate_chromosome_type


class Mutation:

    @staticmethod
    @validate_chromosome_type(type=np.ndarray)
    def flip_bit(sequence: np.ndarray, probability=0.01) -> np.ndarray:
        result = np.asarray(sequence)

        with np.nditer(result, op_flags=['readwrite']) as it:
            for x in it:
                if random.random() < probability:
                    # flip the bit
                    x[...] = 1 - x
                    logging.debug("Mutation probability hit! Mutating gene: %s...", sequence)
        return result

    @staticmethod
    @validate_chromosome_type(type=np.ndarray)
    def swap(sequence: np.ndarray, probability=0.01) -> np.ndarray:
        result = sequence.copy()

        if random.random() < probability:
            swap_pos_1 = random.randint(0, len(sequence) - 1)
            swap_pos_2 = random.randint(0, len(sequence) - 1)
            logging.debug(
                "Mutation probability hit! Mutating gene via swap mutation over positions %d, %d: %s...", swap_pos_1,
                swap_pos_2, sequence)

            result[[swap_pos_1, swap_pos_2]] = result[[swap_pos_2, swap_pos_1]]
        return result

    # TODO implement mutation for real numbers

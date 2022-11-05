import logging
import random

import numpy as np

from src.generic.decorators import validate_chromosome_type


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

    @staticmethod
    @validate_chromosome_type(type=np.ndarray)
    def mutate_real_uniform(sequence: np.ndarray, probability=0.01) -> np.ndarray:
        # Suitable for real numbers
        # TODO simply replace value with another random value
        yield

    @staticmethod
    @validate_chromosome_type(type=np.ndarray)
    def mutate_real_gaussian(sequence: np.ndarray, probability=0.01, use_abs=False) -> np.ndarray:
        """
        Suitable for real numbers
        :param sequence: sequence to mutate
        :param probability: mutation probability
        :param use_abs: use abs() to prevent any negative values after mutation
        :return: mutated sequence
        """
        result = sequence.copy()
        if random.random() < probability:
            logging.debug("Mutation probability hit! Mutating gene: %s...", sequence)
            rng = np.random.default_rng()
            with np.nditer(result, op_flags=['readwrite']) as it:
                for x in it:
                    x[...] = rng.normal(loc=x, scale=abs(x / 10), size=1)[0]
            if use_abs:
                result = np.abs(result)
        return result

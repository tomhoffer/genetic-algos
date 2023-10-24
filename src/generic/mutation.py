import logging
import os
import random

import numpy as np

from src.generic.decorators import validate_chromosome_type


class Mutation:

    @staticmethod
    @validate_chromosome_type(type=np.ndarray)
    def flip_bit(sequence: np.ndarray) -> np.ndarray:
        result = np.asarray(sequence)
        probability = float(os.environ.get("P_MUTATION", default=0.01))

        with np.nditer(result, op_flags=['readwrite']) as it:
            for x in it:
                if random.random() < probability:
                    # flip the bit
                    x[...] = 1 - x
                    logging.debug("Mutation probability hit! Mutating gene: %s...", sequence)
        return result

    @staticmethod
    @validate_chromosome_type(type=np.ndarray)
    def swap(sequence: np.ndarray) -> np.ndarray:
        result = sequence.copy()
        probability = float(os.environ.get("P_MUTATION", default=0.01))

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
    def mutate_real_uniform(sequence: np.ndarray, min: float, max: float, use_abs=True) -> np.ndarray:
        """
        Suitable for real numbers
        :param sequence: sequence to mutate
        :param min: mutated value must be >= min
        :param max: mutated value must be <= max
        :param use_abs: use abs() to prevent any negative values after mutation
        :return: mutated sequence
        """

        if min is not None and max is not None and min > max:
            raise ValueError("Mutation cannot be performed. Min is higher than max! Skipping...")

        result = sequence.copy()
        probability = float(os.environ.get("P_MUTATION", default=0.01))

        if random.random() < probability:
            rng = np.random.default_rng()
            logging.debug("Mutation probability hit! Mutating gene: %s...", sequence)
            chromosome_length = len(sequence)
            mutated_position: int = np.random.randint(low=0, high=chromosome_length, size=1)
            result[mutated_position] = rng.uniform(low=min, high=max, size=1)
            if use_abs:
                result = np.abs(result)
        return result

    @staticmethod
    @validate_chromosome_type(type=np.ndarray)  # TODO implement maximum (return max if number is > max)
    def mutate_real_gaussian(sequence: np.ndarray, use_abs=True, max: float = None, min: float = None) -> np.ndarray:
        """
        Suitable for real numbers
        :param max: Optional: Maximum value to generate
        :param min: Optional: Minimum value to generate
        :param sequence: sequence to mutate
        :param use_abs: use abs() to prevent any negative values after mutation
        :return: mutated sequence
        """

        if min is not None and max is not None and min > max:
            raise ValueError("Mutation cannot be performed. Min is higher than max! Skipping...")

        result = sequence.copy()
        probability = float(os.environ.get("P_MUTATION", default=0.01))
        rng = np.random.default_rng()

        with np.nditer(result, op_flags=['readwrite']) as it:
            for x in it:
                if random.random() < probability:
                    logging.debug("Mutation probability hit! Mutating gene: %s...", sequence)
                    new_val = rng.normal(loc=x, scale=abs(x / 10), size=1)[0]

                    if max and new_val > max:
                        x[...] = max
                        continue

                    if min and new_val < min:
                        x[...] = min
                        continue

                    x[...] = new_val
        if use_abs:
            result = np.abs(result)
        return result

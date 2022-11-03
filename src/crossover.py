import copy
import logging
import random
from typing import Optional, Tuple

import numpy as np

from src.decorators import validate_chromosome_length, validate_parents_chromosome_type
from src.model import Solution


class Crossover:
    @staticmethod
    @validate_parents_chromosome_type(type=np.ndarray)
    @validate_chromosome_length
    def single_point(parent1: Solution, parent2: Solution) -> Optional[Tuple[Solution, Solution]]:
        crossover_pos = random.randint(0, len(parent1.chromosome) - 1)

        offspring_chromosome1 = np.append(parent1.chromosome[:crossover_pos], parent2.chromosome[crossover_pos:])
        offspring_chromosome2 = np.append(parent2.chromosome[:crossover_pos], parent1.chromosome[crossover_pos:])
        logging.debug(
            f"Performing crossover operation over position {crossover_pos} between parents: "
            f"{parent1.chromosome}, {parent2.chromosome}... "
            f"Results: {offspring_chromosome1}, {offspring_chromosome2}"
        )
        return Solution(chromosome=offspring_chromosome1), Solution(chromosome=offspring_chromosome2)

    @classmethod
    def two_point(cls, a: Solution, b: Solution) -> Optional[Tuple[Solution, Solution]]:
        a = copy.deepcopy(a)
        b = copy.deepcopy(b)
        logging.debug(f"Performing 2-point crossover between parents: {a.chromosome}, {b.chromosome}... ")

        for i in range(2):
            a, b = cls.single_point(a, b)
        logging.debug(f"Crossover done, results: {a}, {b}... ")
        return b, a

    @staticmethod
    @validate_parents_chromosome_type(type=np.ndarray)
    @validate_chromosome_length
    def uniform(parent1: Solution, parent2: Solution) -> Optional[Tuple[Solution, Solution]]:
        logging.debug(f"Performing uniform crossover between parents: {parent1.chromosome}, {parent2.chromosome}... ")

        offspring1 = Solution(chromosome=np.empty(0))
        offspring2 = Solution(chromosome=np.empty(0))

        def produce_offspring(a, b):
            return np.asarray([g1 if random.randint(0, 1) else g2 for (g1, g2) in zip(a.chromosome, b.chromosome)])

        offspring1.chromosome = produce_offspring(parent1, parent2)
        offspring2.chromosome = produce_offspring(parent1, parent2)

        return offspring1, offspring2

    # TODO implement blend crossover for real numbers

import copy
import logging
import random
from typing import Optional, Tuple

from src.model import Solution


def crossover_single_point(parent1: Solution, parent2: Solution) -> Optional[Tuple[Solution, Solution]]:
    if len(parent1.chromosome) != len(parent2.chromosome):
        raise ValueError(f"Gene length does not match! Parents: {parent1}, {parent2}")
    crossover_pos = random.randint(0, len(parent1.chromosome) - 1)

    offspring_chromosome1 = parent1.chromosome[:crossover_pos] + parent2.chromosome[crossover_pos:]
    offspring_chromosome2 = parent2.chromosome[:crossover_pos] + parent1.chromosome[crossover_pos:]
    logging.debug(
        f"Performing crossover operation over position {crossover_pos} between parents: "
        f"{parent1.chromosome}, {parent2.chromosome}... "
        f"Results: {offspring_chromosome1}, {offspring_chromosome2}"
    )
    return Solution(chromosome=offspring_chromosome1), Solution(chromosome=offspring_chromosome2)


def crossover_two_point(a: Solution, b: Solution) -> Optional[Tuple[Solution, Solution]]:
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    logging.debug(f"Performing 2-point crossover between parents: {a.chromosome}, {b.chromosome}... ")

    for i in range(2):
        a, b = crossover_single_point(a, b)
    logging.debug(f"Crossover done, results: {a}, {b}... ")
    return b, a

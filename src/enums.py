from enum import Enum

from src.crossover import crossover_single_point, crossover_two_point, crossover_uniform


class CrossoverMethod(Enum):
    SINGLE_POINT = crossover_single_point
    TWO_POINT = crossover_two_point
    UNIFORM = crossover_uniform

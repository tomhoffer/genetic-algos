from enum import Enum

from src.crossover import crossover_single_point, crossover_two_point, crossover_uniform
from src.mutation import mutate_flip_bit, mutate_swap


class CrossoverMethod(Enum):
    SINGLE_POINT = crossover_single_point
    TWO_POINT = crossover_two_point
    UNIFORM = crossover_uniform


class MutationMethod(Enum):
    FLIP_BIT = mutate_flip_bit
    SWAP = mutate_swap

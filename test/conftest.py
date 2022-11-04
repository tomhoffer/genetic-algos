import os
from unittest import mock

import numpy as np
import pytest

from src.model import Population, Solution


def mockenv(**envvars):
    return mock.patch.dict(os.environ, envvars)


def dummy_fn(*args, **kwargs):
    return


def dummy_fitness(chromosome: str):
    return sum(int(x) for x in chromosome)


def initial_population_generator():
    return [Solution(np.asarray([1, 0, 1])) for _ in range(10)]


def dummy_crossover_fn(a, b):
    return a, a


@pytest.fixture
def population_with_zero_fitness():
    return Population(members=[Solution(np.asarray([0, 0, 0])) for _ in range(10)], crossover_fn=dummy_crossover_fn,
                      mutation_fn=dummy_fn, fitness_fn=dummy_fitness, selection_fn=dummy_fn,
                      initial_population_generator_fn=initial_population_generator, population_size=10)


@pytest.fixture
def population_with_identical_solutions():
    return Population(members=[Solution(np.asarray([1, 0, 1]), fitness=1) for i in range(10)],
                      crossover_fn=dummy_crossover_fn,
                      mutation_fn=dummy_fn, fitness_fn=dummy_fitness, selection_fn=dummy_fn,
                      initial_population_generator_fn=initial_population_generator, population_size=10)


@pytest.fixture(scope="class")
def population_with_growing_fitness():
    return Population(members=[Solution(np.asarray([1, i % 2, 1]), fitness=i) for i in range(10)],
                      crossover_fn=dummy_crossover_fn,
                      mutation_fn=dummy_fn, fitness_fn=dummy_fitness, selection_fn=dummy_fn,
                      initial_population_generator_fn=initial_population_generator, population_size=10)


@pytest.fixture
def empty_population():
    return Population(members=[], crossover_fn=dummy_crossover_fn, mutation_fn=dummy_fn, fitness_fn=dummy_fitness,
                      selection_fn=dummy_fn, initial_population_generator_fn=initial_population_generator,
                      population_size=0)

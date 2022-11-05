import os
from unittest import mock

import numpy as np
import pytest

from src.generic.model import Population, Solution


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
                      initial_population_generator_fn=initial_population_generator, population_size=10, elitism=3)


@pytest.fixture
def population_with_identical_solutions():
    return Population(members=[Solution(np.asarray([1, 0, 1]), fitness=1) for i in range(10)],
                      crossover_fn=dummy_crossover_fn,
                      mutation_fn=dummy_fn, fitness_fn=dummy_fitness, selection_fn=dummy_fn,
                      initial_population_generator_fn=initial_population_generator, population_size=10, elitism=3)


@pytest.fixture(scope="class")
def population_with_growing_fitness():
    population = [Solution(np.asarray([0, 0, 0, 0, 0]), fitness=0),
                  Solution(np.asarray([1, 0, 0, 0, 0]), fitness=1),
                  Solution(np.asarray([1, 1, 0, 0, 0]), fitness=2),
                  Solution(np.asarray([1, 1, 1, 0, 0]), fitness=3),
                  Solution(np.asarray([1, 1, 1, 1, 0]), fitness=4),
                  Solution(np.asarray([1, 1, 1, 1, 1]), fitness=5)]

    return Population(members=population,
                      crossover_fn=dummy_crossover_fn,
                      mutation_fn=dummy_fn, fitness_fn=dummy_fitness, selection_fn=dummy_fn,
                      initial_population_generator_fn=initial_population_generator, population_size=10, elitism=3)


@pytest.fixture(scope="class")
def population_with_zero_elitism():
    return Population(members=[Solution(np.asarray([1, 0, 1]), fitness=1) for i in range(10)],
                      crossover_fn=dummy_crossover_fn,
                      mutation_fn=dummy_fn, fitness_fn=dummy_fitness, selection_fn=dummy_fn,
                      initial_population_generator_fn=initial_population_generator, population_size=10, elitism=0)


@pytest.fixture
def empty_population():
    return Population(members=[], crossover_fn=dummy_crossover_fn, mutation_fn=dummy_fn, fitness_fn=dummy_fitness,
                      selection_fn=dummy_fn, initial_population_generator_fn=initial_population_generator,
                      population_size=0, elitism=3)

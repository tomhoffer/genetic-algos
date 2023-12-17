import os
from typing import Callable
from unittest import mock

import numpy as np
import pytest

from src.generic.model import Population, Solution, Hyperparams


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


def dummy_population_validator_fn(members):
    return True


@pytest.fixture
def population_with_zero_fitness():
    return [Solution(np.asarray([0, 0, 0])) for _ in range(10)]


@pytest.fixture
def population_with_ninf_fitness():
    return [Solution(np.asarray([0, 0, 0]), fitness=np.NINF) for _ in range(10)]


@pytest.fixture
def population_with_identical_solutions():
    return [Solution(np.asarray([1, 0, 1]), fitness=1) for i in range(10)]


@pytest.fixture
def population_with_growing_fitness():
    return [Solution(np.asarray([0, 0, 0, 0, 0]), fitness=0),
            Solution(np.asarray([1, 0, 0, 0, 0]), fitness=1),
            Solution(np.asarray([1, 1, 0, 0, 0]), fitness=2),
            Solution(np.asarray([1, 1, 1, 0, 0]), fitness=3),
            Solution(np.asarray([1, 1, 1, 1, 0]), fitness=4),
            Solution(np.asarray([1, 1, 1, 1, 1]), fitness=5)]


@pytest.fixture
def empty_population():
    return []


@pytest.fixture
def configurable_hyperparams(request):
    config = {
        'elitism': request.node.get_closest_marker("elitism"),
        'fitness_fn': request.node.get_closest_marker("fitness_fn"),
        'mutation_fn': request.node.get_closest_marker("mutation_fn"),
        'crossover_fn': request.node.get_closest_marker("crossover_fn"),
        'stopping_criteria_fn': request.node.get_closest_marker("stopping_criteria_fn"),
        'selection_fn': request.node.get_closest_marker("selection_fn"),
        'chromosome_validator_fn': request.node.get_closest_marker("chromosome_validator_fn"),
        'initial_population_generator_fn': request.node.get_closest_marker("initial_population_generator_fn"),
        'population_size': request.node.get_closest_marker("population_size"),
    }

    # Fill in dummy attributes if they were not specified
    for key, value in config.items():
        if value is None:
            if 'fn' in key:
                config[key] = dummy_fn
            else:
                config[key] = 0
        else:
            config[key] = config[key].args[0]

    return Hyperparams(elitism=config['elitism'], fitness_fn=config['fitness_fn'], mutation_fn=config['mutation_fn'],
                       crossover_fn=config['crossover_fn'],
                       stopping_criteria_fn=config['stopping_criteria_fn'], selection_fn=config['selection_fn'],
                       chromosome_validator_fn=config['chromosome_validator_fn'],
                       initial_population_generator_fn=config['initial_population_generator_fn'],
                       population_size=config['population_size'])


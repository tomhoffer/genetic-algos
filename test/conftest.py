import pytest

from src.model import Population, Solution


def dummy_fn():
    return


@pytest.fixture
def population_with_zero_fitness():
    return Population(members=[Solution('101') for _ in range(10)], crossover_fn=dummy_fn,
                      mutation_fn=dummy_fn, fitness_fn=dummy_fn, selection_fn=dummy_fn,
                      initial_population_generator_fn=dummy_fn, population_size=10)


@pytest.fixture
def population_with_positive_fitness():
    return Population(members=[Solution('101', fitness=1) for _ in range(10)], crossover_fn=dummy_fn,
                      mutation_fn=dummy_fn, fitness_fn=dummy_fn, selection_fn=dummy_fn,
                      initial_population_generator_fn=dummy_fn, population_size=10)


@pytest.fixture
def empty_population():
    return Population(members=[], crossover_fn=dummy_fn, mutation_fn=dummy_fn, fitness_fn=dummy_fn,
                      selection_fn=dummy_fn, initial_population_generator_fn=dummy_fn, population_size=0)

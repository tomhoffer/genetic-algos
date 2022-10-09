from typing import List

import pytest

from src.model import Population, Solution
from src.selection import select_tournament


def dummy_fn():
    return


@pytest.fixture
def population_with_zero_fitness():
    return Population(members=[Solution('101') for _ in range(10)], crossover_fn=dummy_fn,
                      mutation_fn=dummy_fn, fitness_fn=dummy_fn, selection_fn=dummy_fn,
                      initial_population_generator_fn=dummy_fn)


@pytest.fixture
def population_with_positive_fitness():
    return Population(members=[Solution('101', fitness=1) for _ in range(10)], crossover_fn=dummy_fn,
                      mutation_fn=dummy_fn, fitness_fn=dummy_fn, selection_fn=dummy_fn,
                      initial_population_generator_fn=dummy_fn)


@pytest.fixture
def empty_population():
    return Population(members=[], crossover_fn=dummy_fn, mutation_fn=dummy_fn, fitness_fn=dummy_fn,
                      selection_fn=dummy_fn, initial_population_generator_fn=dummy_fn)


def test_empty_population(empty_population):
    with pytest.raises(ValueError) as err:
        select_tournament(empty_population)
    assert str(err.value) == "Population size lower than 1! Actual: 0"


@pytest.mark.parametrize("size", [3, 5, 10])
def test_tournament_size(size, population_with_zero_fitness):
    # Always return number of offsprings equal to population size, no matter the tournament size
    result: List[Solution] = select_tournament(population=population_with_zero_fitness, tournament_size=size)
    assert len(result) == len(population_with_zero_fitness.members)


def test_all_zero_fitness(population_with_zero_fitness):
    # Selection works even when all Solutions have fitness == 0
    result: List[Solution] = select_tournament(population=population_with_zero_fitness)
    assert len(result) == len(population_with_zero_fitness.members)
    assert result == population_with_zero_fitness.members


def test_all_identical(population_with_positive_fitness):
    # Selection works for population with identical positive fitness values
    result: List[Solution] = select_tournament(population=population_with_positive_fitness)
    assert result == population_with_positive_fitness.members

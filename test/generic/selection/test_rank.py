import copy
from typing import List

import pytest

from src.generic.model import Solution, Population
from src.generic.selection import Selection
from conftest import mockenv


def test_empty_population(empty_population):
    with pytest.raises(ValueError) as err:
        Selection.rank(empty_population)
    assert str(err.value) == "Population size lower than 1! Actual: 0"


def test_all_zero_fitness(population_with_zero_fitness):
    # Selection works even when all Solutions have fitness == 0
    result: List[Solution] = Selection.rank(population=population_with_zero_fitness)
    assert len(result) == len(population_with_zero_fitness.members)
    assert result == population_with_zero_fitness.members


def test_all_ninf_fitness(population_with_ninf_fitness):
    # Selection works even when all Solutions have fitness == np.NINF
    result: List[Solution] = Selection.rank(population=population_with_ninf_fitness)
    assert len(result) == len(population_with_ninf_fitness.members)
    assert result == population_with_ninf_fitness.members


def test_all_identical(population_with_identical_solutions):
    # Selection works for population with identical positive fitness values
    result: List[Solution] = Selection.rank(population=population_with_identical_solutions)
    assert result == population_with_identical_solutions.members


@mockenv(ELITISM="True")
def test_elitism_enabled(configurable_population):
    # Elitism is effective when enabled
    population = configurable_population(elitism=3)
    old_population: Population = copy.deepcopy(population)
    result: List[Solution] = Selection.rank(population=population)

    elite_individuals = sorted(old_population.members, key=lambda x: x.fitness, reverse=True)[:3]
    assert len(result) == len(old_population.members)

    for elite in elite_individuals:
        assert elite in result
        assert result.count(elite) == 1


@mockenv(ELITISM="True")
def test_elitism_zero(configurable_population, mocker):
    # Elitism is not effective when enabled but set to 0
    mocked_find_best = mocker.patch('src.generic.selection._find_best')
    population = configurable_population(elitism=0)
    result: List[Solution] = Selection.rank(population=population)
    assert mocked_find_best.call_count == 0


@mockenv(ELITISM="False")
def test_elitism_disabled(configurable_population, mocker):
    mocked_find_best = mocker.patch('src.generic.selection._find_best')
    population = configurable_population(elitism=3)
    result: List[Solution] = Selection.rank(population=population)
    assert mocked_find_best.call_count == 0

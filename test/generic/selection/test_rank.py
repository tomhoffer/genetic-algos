import copy
from typing import List

import pytest

from src.generic.model import Solution
from src.generic.selection import Selection
from conftest import mockenv


def test_empty_population(empty_population):
    with pytest.raises(ValueError) as err:
        Selection.rank(empty_population)
    assert str(err.value) == "Population size lower than 1! Actual: 0"


def test_all_zero_fitness(population_with_zero_fitness):
    # Selection works even when all Solutions have fitness == 0
    result: List[Solution] = Selection.rank(population=population_with_zero_fitness)
    assert len(result) == len(population_with_zero_fitness)
    assert result == population_with_zero_fitness


def test_all_ninf_fitness(population_with_ninf_fitness):
    # Selection works even when all Solutions have fitness == np.NINF
    result: List[Solution] = Selection.rank(population=population_with_ninf_fitness)
    assert len(result) == len(population_with_ninf_fitness)
    assert result == population_with_ninf_fitness


def test_all_identical(population_with_identical_solutions):
    # Selection works for population with identical positive fitness values
    result: List[Solution] = Selection.rank(population=population_with_identical_solutions)
    assert result == population_with_identical_solutions


@mockenv(ELITISM="3")
def test_elitism_enabled(population_with_growing_fitness):
    # Elitism is effective when set to value > 3
    old_population: List[Solution] = copy.deepcopy(population_with_growing_fitness)
    result: List[Solution] = Selection.rank(population=population_with_growing_fitness)

    elite_individuals = sorted(old_population, key=lambda x: x.fitness, reverse=True)[:3]
    assert len(result) == len(old_population)

    for elite in elite_individuals:
        assert elite in result
        assert result.count(elite) == 1


@mockenv(ELITISM="0")
def test_elitism_zero(population_with_growing_fitness, mocker):
    # Elitism is not effective when set to 0
    mocked_find_best = mocker.patch('src.generic.selection._find_best')
    Selection.rank(population=population_with_growing_fitness)
    assert mocked_find_best.call_count == 0

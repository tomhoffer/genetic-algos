import copy
from typing import List

import pytest

from src.generic.model import Solution, Population
from src.generic.selection import Selection
from conftest import mockenv


def test_empty_population(empty_population):
    with pytest.raises(ValueError) as err:
        Selection.tournament(empty_population)
    assert str(err.value) == "Population size lower than 1! Actual: 0"


@pytest.mark.parametrize("size", [3, 5, 10])
def test_tournament_size(size, population_with_zero_fitness):
    # Always return number of offsprings equal to population size, no matter the tournament size
    result: List[Solution] = Selection.tournament(population=population_with_zero_fitness, tournament_size=size)
    assert len(result) == len(population_with_zero_fitness.members)


def test_all_zero_fitness(population_with_zero_fitness):
    # Selection works even when all Solutions have fitness == 0
    result: List[Solution] = Selection.tournament(population=population_with_zero_fitness)
    assert len(result) == len(population_with_zero_fitness.members)
    assert result == population_with_zero_fitness.members


def test_all_identical(population_with_identical_solutions):
    # Selection works for population with identical positive fitness values
    result: List[Solution] = Selection.tournament(population=population_with_identical_solutions)
    assert result == population_with_identical_solutions.members


@mockenv(ELITISM="True")
def test_elitism_enabled(population_with_growing_fitness):
    # Elitism ensures best individuals survive into next generation
    old_population: Population = copy.deepcopy(population_with_growing_fitness)
    result: List[Solution] = Selection.tournament(population=population_with_growing_fitness)

    elite_individuals = sorted(old_population.members, key=lambda x: x.fitness, reverse=True)[:3]
    assert len(result) == len(old_population.members)
    for elite in elite_individuals:
        assert elite in result


@mockenv(ELITISM="True")
def test_elitism_enabled(population_with_zero_elitism):
    # Elitism is not effective when enabled but set to 0
    old_population: Population = copy.deepcopy(population_with_zero_elitism)
    result: List[Solution] = Selection.tournament(population=population_with_zero_elitism)

    elite_individuals = sorted(old_population.members, key=lambda x: x.fitness, reverse=True)[:3]
    assert len(result) == len(old_population.members)
    for elite in elite_individuals:
        assert elite in result

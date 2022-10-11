from typing import List

import pytest

from src.model import Solution
from src.selection import Selection


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


def test_all_identical(population_with_positive_fitness):
    # Selection works for population with identical positive fitness values
    result: List[Solution] = Selection.tournament(population=population_with_positive_fitness)
    assert result == population_with_positive_fitness.members

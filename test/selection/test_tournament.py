import pytest

from src.model import Population
from src.selection import select_tournament


def dummy_fn():
    return


def test_empty_population():
    p = Population(members=[], crossover_fn=dummy_fn, mutation_fn=dummy_fn, fitness_fn=dummy_fn)
    with pytest.raises(ValueError) as err:
        result = select_tournament(p)

    assert str(err.value) == "Population size lower than 1! Actual: 0"


@pytest.mark.parametrize("given,expected", [("3+5", 8), ("2+4", 6), ("6*9", 42)]) # TODO finish
def test_tournament(given, expected):
    p = Population(members=[], crossover_fn=dummy_fn, mutation_fn=dummy_fn, fitness_fn=dummy_fn)

import pytest

from src.crossover import Crossover
from src.model import Solution


def test_uniform(mocker):
    mocker.patch("random.randint", return_value=1)
    a = Solution("0000")
    b = Solution("1111")
    res1, res2 = Crossover.uniform(a, b)

    assert res1 is not None
    assert res2 is not None

    assert len(res1.chromosome) == 4
    assert len(res2.chromosome) == 4

    assert type(res1.chromosome) is str
    assert type(res2.chromosome) is str


def test_different_chromosome_length():
    with pytest.raises(ValueError) as err:
        Crossover.uniform(Solution("0"), Solution("000"))
    assert str(
        err.value) == "Gene length does not match! Parents: Solution(chromosome='0', fitness=0), Solution(chromosome='000', fitness=0)"

import pytest

from src.crossover import crossover_single_point
from src.model import Solution


@pytest.mark.parametrize("position, expected",
                         [(0, (Solution("11111111"), Solution("00000000"))),
                          (4, (Solution("00001111"), Solution("11110000"))),
                          (8, (Solution("00000000"), Solution("11111111")))])
def test_single_point_even_length(mocker, position, expected):
    # Test that crossover works over given position
    mocker.patch("random.randint", return_value=position)
    a = Solution("00000000")
    b = Solution("11111111")
    assert crossover_single_point(a, b) == expected


@pytest.mark.parametrize("position, expected",
                         [(0, (Solution("11111"), Solution("00000"))),
                          (3, (Solution("00011"), Solution("11100"))),
                          (5, (Solution("00000"), Solution("11111")))])
def test_single_point_odd_length(mocker, position, expected):
    # Test that crossover works over given position
    mocker.patch("random.randint", return_value=position)
    a = Solution("00000")
    b = Solution("11111")
    assert crossover_single_point(a, b) == expected


def test_different_chromosome_length():
    with pytest.raises(ValueError) as err:
        crossover_single_point(Solution("0"), Solution("000"))
    assert str(err.value) == "Gene length does not match! Parents: Solution(chromosome='0', fitness=0), Solution(chromosome='000', fitness=0)"
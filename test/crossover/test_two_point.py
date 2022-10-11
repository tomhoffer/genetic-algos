import pytest

from src.crossover import Crossover
from src.model import Solution


@pytest.mark.parametrize("pos1, pos2, expected",
                         [(0, 0, (Solution("11111111"), Solution("00000000"))),
                          (0, 4, (Solution("00001111"), Solution("11110000"))),
                          (4, 6, (Solution("11110011"), Solution("00001100"))),
                          (8, 8, (Solution("11111111"), Solution("00000000")))])
def test_two_point_even_length(mocker, pos1, pos2, expected):
    # Test that crossover works over given position
    mocker.patch("random.randint", side_effect=[pos1, pos2])
    a = Solution("00000000")
    b = Solution("11111111")
    assert Crossover.two_point(a, b) == expected


@pytest.mark.parametrize("pos1, pos2, expected",
                         [(0, 0, (Solution("1111111"), Solution("0000000"))),
                          (0, 3, (Solution("0001111"), Solution("1110000"))),
                          (3, 6, (Solution("1110001"), Solution("0001110"))),
                          (7, 7, (Solution("1111111"), Solution("0000000")))])
def test_two_point_odd_length(mocker, pos1, pos2, expected):
    # Test that crossover works over given position
    mocker.patch("random.randint", side_effect=[pos1, pos2])
    a = Solution("0000000")
    b = Solution("1111111")
    assert Crossover.two_point(a, b) == expected

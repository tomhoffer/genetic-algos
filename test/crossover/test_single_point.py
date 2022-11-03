import numpy as np
import pytest

from src.crossover import Crossover
from src.model import Solution


@pytest.mark.parametrize("position, expected",
                         [(0, (Solution(np.asarray([1, 1, 1, 1, 1, 1, 1, 1])),
                               Solution(np.asarray([0, 0, 0, 0, 0, 0, 0, 0])))),
                          (4, (Solution(np.asarray([0, 0, 0, 0, 1, 1, 1, 1])),
                               Solution(np.asarray([1, 1, 1, 1, 0, 0, 0, 0])))),
                          (8, (Solution(np.asarray([0, 0, 0, 0, 0, 0, 0, 0])),
                               Solution(np.asarray([1, 1, 1, 1, 1, 1, 1, 1]))))
                          ])
def test_single_point_even_length(mocker, position, expected):
    # Test that crossover works over given position
    mocker.patch("random.randint", return_value=position)
    a = Solution(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))
    b = Solution(np.asarray([1, 1, 1, 1, 1, 1, 1, 1]))

    res1, res2 = Crossover.single_point(a, b)
    np.testing.assert_array_equal(res1.chromosome, expected[0].chromosome)
    np.testing.assert_array_equal(res2.chromosome, expected[1].chromosome)


@pytest.mark.parametrize("position, expected",
                         [(0, (Solution(np.asarray([1, 1, 1, 1, 1])), Solution(np.asarray([0, 0, 0, 0, 0])))),
                          (3, (Solution(np.asarray([0, 0, 0, 1, 1])), Solution(np.asarray([1, 1, 1, 0, 0])))),
                          (5, (Solution(np.asarray([0, 0, 0, 0, 0])), Solution(np.asarray([1, 1, 1, 1, 1]))))])
def test_single_point_odd_length(mocker, position, expected):
    # Test that crossover works over given position
    mocker.patch("random.randint", return_value=position)
    a = Solution(np.asarray([0, 0, 0, 0, 0]))
    b = Solution(np.asarray([1, 1, 1, 1, 1]))
    res1, res2 = Crossover.single_point(a, b)
    np.testing.assert_array_equal(res1.chromosome, expected[0].chromosome)
    np.testing.assert_array_equal(res2.chromosome, expected[1].chromosome)


def test_different_chromosome_length():
    with pytest.raises(ValueError, match=r"Gene length does not match! Parents: .*") as err:
        Crossover.single_point(Solution(np.asarray([0])), Solution(np.asarray([1, 0])))


def test_invalid_chromosome_type():
    with pytest.raises(TypeError, match=r"Parent chromosomes do not match required type: .*\. Parents: .*, .*") as err:
        Crossover.uniform(Solution(0.98), Solution("000"))

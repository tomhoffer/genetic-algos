import numpy as np
import pytest

from src.generic.crossover import Crossover
from src.generic.model import Solution


@pytest.mark.parametrize("pos1, pos2, expected",
                         [(0, 0, (Solution(np.asarray([1, 1, 1, 1, 1, 1, 1, 1])),
                                  Solution(np.asarray([0, 0, 0, 0, 0, 0, 0, 0])))),
                          (0, 4, (Solution(np.asarray([0, 0, 0, 0, 1, 1, 1, 1])),
                                  Solution(np.asarray([1, 1, 1, 1, 0, 0, 0, 0])))),
                          (4, 6, (Solution(np.asarray([1, 1, 1, 1, 0, 0, 1, 1])),
                                  Solution(np.asarray([0, 0, 0, 0, 1, 1, 0, 0])))),
                          (8, 8, (Solution(np.asarray([1, 1, 1, 1, 1, 1, 1, 1])),
                                  Solution(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))))])
def test_two_point_even_length(mocker, pos1, pos2, expected):
    # Test that crossover works over given position
    mocker.patch("random.randint", side_effect=[pos1, pos2])
    a = Solution(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))
    b = Solution(np.asarray([1, 1, 1, 1, 1, 1, 1, 1]))
    res1, res2 = Crossover.two_point(a, b)
    np.testing.assert_array_equal(res1.chromosome, expected[0].chromosome)
    np.testing.assert_array_equal(res2.chromosome, expected[1].chromosome)


@pytest.mark.parametrize("pos1, pos2, expected",
                         [(0, 0,
                           (Solution(np.asarray([1, 1, 1, 1, 1, 1, 1])), Solution(np.asarray([0, 0, 0, 0, 0, 0, 0])))),
                          (0, 3,
                           (Solution(np.asarray([0, 0, 0, 1, 1, 1, 1])), Solution(np.asarray([1, 1, 1, 0, 0, 0, 0])))),
                          (3, 6,
                           (Solution(np.asarray([1, 1, 1, 0, 0, 0, 1])), Solution(np.asarray([0, 0, 0, 1, 1, 1, 0])))),
                          (7, 7,
                           (Solution(np.asarray([1, 1, 1, 1, 1, 1, 1])), Solution(np.asarray([0, 0, 0, 0, 0, 0, 0]))))])
def test_two_point_odd_length(mocker, pos1, pos2, expected):
    # Test that crossover works over given position
    mocker.patch("random.randint", side_effect=[pos1, pos2])
    a = Solution(np.asarray([0, 0, 0, 0, 0, 0, 0]))
    b = Solution(np.asarray([1, 1, 1, 1, 1, 1, 1]))
    res1, res2 = Crossover.two_point(a, b)
    np.testing.assert_array_equal(res1.chromosome, expected[0].chromosome)
    np.testing.assert_array_equal(res2.chromosome, expected[1].chromosome)


def test_two_point_2d(mocker):
    # Test that crossover works also for 2d arrays
    mocker.patch("random.randint", side_effect=[0, 2])
    a = Solution(np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    b = Solution(np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]))

    res1, res2 = Crossover.two_point(a, b)
    np.testing.assert_array_equal(res1.chromosome, np.asarray([[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]]))
    np.testing.assert_array_equal(res2.chromosome, np.asarray([[1, 1, 1], [1, 1, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]))

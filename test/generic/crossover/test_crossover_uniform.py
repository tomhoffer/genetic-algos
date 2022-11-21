import numpy as np
import pytest

from src.generic.crossover import Crossover
from src.generic.model import Solution


def test_uniform(mocker):
    mocker.patch("random.randint", return_value=1)
    a = Solution(np.array([0, 0, 0, 0]))
    b = Solution(np.array([1, 1, 1, 1]))
    res1, res2 = Crossover.uniform(a, b)

    assert res1.chromosome.shape == res2.chromosome.shape == a.chromosome.shape == b.chromosome.shape


def test_uniform_2d():
    # Test that crossover works also for 2d arrays
    a = Solution(np.asarray([[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]))
    b = Solution(np.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]]))

    res1, res2 = Crossover.uniform(a, b)
    assert res1.chromosome.shape == res2.chromosome.shape == a.chromosome.shape == b.chromosome.shape


def test_different_chromosome_length():
    with pytest.raises(ValueError, match=r"Gene length does not match! Parents: .*") as err:
        Crossover.uniform(Solution(np.asarray([0])), Solution(np.asarray([0, 1, 0])))


def test_invalid_chromosome_type():
    with pytest.raises(TypeError, match=r"Parent chromosomes do not match required type: .*\. Parents: .*, .*") as err:
        Crossover.uniform(Solution(0.98), Solution("000"))

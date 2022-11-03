import numpy as np
import pytest

from src.crossover import Crossover
from src.model import Solution


def test_uniform(mocker):
    mocker.patch("random.randint", return_value=1)
    a = Solution(np.array([0, 0, 0, 0]))
    b = Solution(np.array([1, 1, 1, 1]))
    res1, res2 = Crossover.uniform(a, b)

    assert res1 is not None
    assert res2 is not None

    assert len(res1.chromosome) == 4
    assert len(res2.chromosome) == 4

    assert type(res1.chromosome) is np.ndarray
    assert type(res2.chromosome) is np.ndarray


def test_different_chromosome_length():
    with pytest.raises(ValueError, match=r"Gene length does not match! Parents: .*") as err:
        Crossover.uniform(Solution(np.asarray([0])), Solution(np.asarray([0, 1, 0])))


def test_invalid_chromosome_type():
    with pytest.raises(TypeError, match=r"Parent chromosomes do not match required type: .*\. Parents: .*, .*") as err:
        Crossover.uniform(Solution(0.98), Solution("000"))

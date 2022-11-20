import numpy as np
import pytest

from src.generic.mutation import Mutation
from conftest import mockenv


@mockenv(P_MUTATION="1.0")
def test_probability_100():
    res = Mutation.flip_bit(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(res, np.asarray([1, 1, 1, 1, 1, 1, 1, 1]))


@mockenv(P_MUTATION="0")
def test_probability_0():
    res = Mutation.flip_bit(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(res, np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))


def test_invalid_chromosome_type():
    invalid_values = [123, "123", 1.23]
    for val in invalid_values:
        with pytest.raises(TypeError, match=r"Chromosome does not match required type: .*\. Chromosome: .*") as err:
            Mutation.flip_bit(val, probability=1)

import numpy as np
import pytest

from src.generic.mutation import Mutation
from conftest import mockenv


@mockenv(P_MUTATION="1.0")
def test_probability_100():
    before = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
    after = Mutation.flip_bit(before)
    np.testing.assert_array_equal(before, after)
    assert before.shape == after.shape


@mockenv(P_MUTATION="0")
def test_probability_0():
    before = np.asarray([0, 0, 0, 0, 0, 0, 0, 0])
    after = Mutation.flip_bit(before)
    np.testing.assert_array_equal(before, after)


@mockenv(P_MUTATION="1.0")
def test_invalid_chromosome_type():
    invalid_values = [123, "123", 1.23]
    for val in invalid_values:
        with pytest.raises(TypeError, match=r"Chromosome does not match required type: .*\. Chromosome: .*") as err:
            Mutation.flip_bit(val)

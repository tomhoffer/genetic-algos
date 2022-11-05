import numpy as np
import pytest
from numpy.testing import assert_raises, assert_array_equal

from src.generic.mutation import Mutation


def test_probability_100():
    before = np.asarray([1.27, 3.8, 0.4])
    after = Mutation.mutate_real_gaussian(before, probability=1.0)
    assert_raises(AssertionError, assert_array_equal, before, after)


def test_probability_0():
    before = np.asarray([1.27, 3.8, 0.4])
    after = Mutation.mutate_real_gaussian(before, probability=0)
    assert_array_equal(before, after)


def test_use_absolute():
    before = np.asarray([-1.27, 3.8, -0.4, 0])
    after = Mutation.mutate_real_gaussian(before, probability=1.0, use_abs=True)
    for x in after:
        assert x >= 0


def test_invalid_chromosome_type():
    invalid_values = [123, "123", 1.23]
    for val in invalid_values:
        with pytest.raises(TypeError, match=r"Chromosome does not match required type: .*\. Chromosome: .*") as err:
            Mutation.mutate_real_gaussian(val, probability=1)

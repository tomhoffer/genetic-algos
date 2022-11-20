import numpy as np
import pytest
from numpy.testing import assert_raises, assert_array_equal

from src.generic.mutation import Mutation
from conftest import mockenv


@mockenv(P_MUTATION="1.0")
def test_probability_100():
    before = np.asarray([1.27, 3.8, 0.4])
    after = Mutation.mutate_real_uniform(before, min=1, max=10)
    assert_raises(AssertionError, assert_array_equal, before, after)


@mockenv(P_MUTATION="1.0")
def test_min_max():
    before = np.asarray([1.27, 3.8, 0.4])
    after = Mutation.mutate_real_uniform(before, min=1, max=10)
    assert np.all((after >= 1) & (after <= 10))


@mockenv(P_MUTATION="0")
def test_probability_0():
    before = np.asarray([1.27, 3.8, 0.4])
    after = Mutation.mutate_real_uniform(before, min=1, max=10)
    assert_array_equal(before, after)


@mockenv(P_MUTATION="1.0")
def test_invalid_chromosome_type():
    invalid_values = [123, "123", 1.23]
    for val in invalid_values:
        with pytest.raises(TypeError, match=r"Chromosome does not match required type: .*\. Chromosome: .*") as err:
            Mutation.mutate_real_uniform(val)

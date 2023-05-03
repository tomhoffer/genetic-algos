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
    assert before.shape == after.shape


@mockenv(P_MUTATION="1.0")
@pytest.mark.parametrize("min, max", [(1, 5), (1, 1), (5, 5)])
def test_min_max(min, max, mocker):
    mocker.patch("numpy.random.uniform", return_value=3)
    before = np.asarray([1, 2, 3, 4, 5])
    after = Mutation.mutate_real_uniform(before, min=min, max=max)
    assert np.all((after >= min) & (after <= max))


@mockenv(P_MUTATION="1.0")
def test_min_max_invalid():
    # Minimum > Maximum
    before = np.asarray([1, 2, 3, 4, 5])
    with pytest.raises(ValueError) as err:
        Mutation.mutate_real_uniform(before, min=1, max=0)
    assert str(err.value) == "Mutation cannot be performed. Min is higher than max! Skipping..."


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
            Mutation.mutate_real_uniform(val, min=0, max=1)


@mockenv(P_MUTATION="1.0")
def test_use_absolute():
    before = np.asarray([-1.27, 3.8, -0.4, 0])
    after = Mutation.mutate_real_uniform(before, use_abs=True, min=0, max=5)
    for x in after:
        assert x >= 0
    assert before.shape == after.shape

import numpy as np
import pytest

from src.generic.mutation import Mutation
from conftest import mockenv


@mockenv(P_MUTATION="0")
def test_probability_0():
    # Test that chromosome has NOT been mutated
    res = Mutation.swap(np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))
    np.testing.assert_array_equal(res, np.asarray([0, 0, 0, 0, 0, 0, 0, 0]))


@pytest.mark.parametrize("pos1, pos2, expected",
                         [
                             (0, 0, np.asarray([0, 1, 2, 3, 4])),
                             (0, 2, np.asarray([2, 1, 0, 3, 4])),
                             (0, 4, np.asarray([4, 1, 2, 3, 0])),
                             (4, 4, np.asarray([0, 1, 2, 3, 4])),
                         ])
@mockenv(P_MUTATION="1.0")
def test_probability_1(mocker, pos1, pos2, expected):
    # Test that chromosome HAS been correctly mutated
    mocker.patch("random.randint", side_effect=[pos1, pos2])
    res = Mutation.swap(np.asarray([0, 1, 2, 3, 4]))
    np.testing.assert_array_equal(res, expected)


@mockenv(P_MUTATION="1.0")
def test_invalid_chromosome_type():
    invalid_values = [123, "123", 1.23]
    for val in invalid_values:
        with pytest.raises(TypeError, match=r"Chromosome does not match required type: .*\. Chromosome: .*") as err:
            Mutation.swap(val)

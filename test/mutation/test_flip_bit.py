import pytest

from src.mutation import Mutation


def test_probability_100():
    assert Mutation.flip_bit("0000000000", probability=1.0) == "1111111111"


def test_probability_0():
    assert Mutation.flip_bit("0000000000", probability=0) == "0000000000"


def test_invalid_chromosome_type():
    with pytest.raises(TypeError, match=r"Chromosome does not match required type: .*\. Chromosome: .*") as err:
        Mutation.flip_bit(chromosome=123, probability=1)

from src.mutation import Mutation


def test_probability_100():
    assert Mutation.flip_bit("0000000000", probability=1.0) == "1111111111"


def test_probability_0():
    assert Mutation.flip_bit("0000000000", probability=0) == "0000000000"

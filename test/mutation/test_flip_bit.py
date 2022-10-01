from src.mutation import mutate_flip_bit


def test_probability_100():
    assert mutate_flip_bit("0000000000", probability=1.0) == "1111111111"


def test_probability_0():
    assert mutate_flip_bit("0000000000", probability=0) == "0000000000"

from src.mutation import mutate


def test_probability_100():
    assert mutate("0000000000", probability=1.0) == "1111111111"


def test_probability_0():
    assert mutate("0000000000", probability=0) == "0000000000"

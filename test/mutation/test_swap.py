import pytest

from src.mutation import mutate_swap


def test_probability_0():
    # Test that chromosome has NOT been mutated
    assert mutate_swap("0123", probability=0) == "0123"


@pytest.mark.parametrize("pos1, pos2, expected",
                         [
                             (0, 0, "01234"),
                             (0, 2, "21034"),
                             (0, 4, "41230"),
                             (4, 4, "01234"),
                         ])
def test_probability_1(mocker, pos1, pos2, expected):
    # Test that chromosome HAS been correctly mutated
    mocker.patch("random.randint", side_effect=[pos1, pos2])
    assert mutate_swap("01234", probability=1) == expected

import pytest

from src.mutation import Mutation


def test_probability_0():
    # Test that chromosome has NOT been mutated
    assert Mutation.swap("0123", probability=0) == "0123"


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
    assert Mutation.swap("01234", probability=1) == expected


def test_invalid_chromosome_type():
    with pytest.raises(TypeError, match=r"Chromosome does not match required type: .*\. Chromosome: .*") as err:
        Mutation.swap(123, probability=1)
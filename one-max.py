import random
from typing import List

POPULATION_SIZE = 50
STR_LEN = 100
P_MUTATION = 0.01


def generate_initial_population(size: int = POPULATION_SIZE) -> List[str]:
    result = []
    for _ in range(size):
        el = ""
        for _ in range(STR_LEN):
            el = el + str(random.randint(0, 1))
        result.append(el)
    return result


def fitness(element: str) -> int:
    sum = 0
    for bit in element:
        sum += int(bit)
    return sum


def select(population: List[str]):
    
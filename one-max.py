import logging
import random
from typing import List

from src.conf import CONFIG
from src.enums import CrossoverMethod, MutationMethod, SelectionMethod
from src.executor import ParallelExecutor
from src.hyperparams import Hyperparams
from src.model import Solution

POPULATION_SIZE = 10
N_PROCESSES = 5
TIMEOUT_SECONDS = 120

logging.basicConfig(level=logging.DEBUG)


def initial_population_generator() -> List[Solution]:
    result: List[Solution] = []
    for _ in range(POPULATION_SIZE):
        el = ""
        for _ in range(CONFIG["STR_LEN"]):
            el = el + str(random.randint(0, 1))
        result.append(Solution(chromosome=el))
    return result


def fitness(chromosome: str) -> int:
    return sum(int(x) for x in chromosome)


if __name__ == "__main__":
    """
    ParallelExecutor.run((Hyperparams(crossover_fn=CrossoverMethod.TWO_POINT,
                                      initial_population_generator_fn=initial_population_generator,
                                      mutation_fn=MutationMethod.FLIP_BIT, selection_fn=SelectionMethod.TOURNAMENT,
                                      fitness_fn=fitness), 0))

    exit(0)
    """

    ParallelExecutor.run_parallel(Hyperparams(crossover_fn=CrossoverMethod.TWO_POINT,
                                              initial_population_generator_fn=initial_population_generator,
                                              mutation_fn=MutationMethod.FLIP_BIT,
                                              selection_fn=SelectionMethod.TOURNAMENT,
                                              fitness_fn=fitness))

    exit(0)

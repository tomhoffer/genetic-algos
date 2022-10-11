import logging
import random
from typing import List

from src.conf import CONFIG
from src.crossover import Crossover
from src.hyperparams import HyperparamEvaluator
from src.model import Solution
from src.mutation import Mutation
from src.selection import Selection

POPULATION_SIZE = 10
N_PROCESSES = 2
TIMEOUT_SECONDS = 120

logging.basicConfig(level=logging.INFO)


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
    TrainingExecutor.run((Hyperparams(crossover_fn=Crossover.two_point,
                                      initial_population_generator_fn=initial_population_generator,
                                      mutation_fn=Mutation.flip_bit, selection_fn=Selection.tournament,
                                      fitness_fn=fitness), 0))

    exit(0)

    """

    """
    TrainingExecutor.run_parallel(Hyperparams(crossover_fn=Crossover.two_point,
                                              initial_population_generator_fn=initial_population_generator,
                                              mutation_fn=Mutation.flip_bit,
                                              selection_fn=Selection.tournament,
                                              fitness_fn=fitness))

    exit(0)
    """

    selection_methods = [Selection.tournament, Selection.roulette, Selection.rank]
    crossover_methods = [Crossover.two_point, Crossover.single_point, Crossover.uniform]
    mutation_methods = [Mutation.flip_bit, Mutation.swap]
    population_sizes = [10, 100, 200]

    evaluator = HyperparamEvaluator(selection_methods=selection_methods, mutation_methods=mutation_methods,
                                    crossover_methods=crossover_methods, population_sizes=population_sizes,
                                    fitness_fn=fitness, initial_population_generation_fn=initial_population_generator)

    evaluator.grid_search()

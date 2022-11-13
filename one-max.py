import logging
import os
from typing import List

import numpy as np
from dotenv import load_dotenv

from src.generic.crossover import Crossover
from src.generic.executor import TrainingExecutor
from src.generic.model import Solution, Hyperparams
from src.generic.mutation import Mutation
from src.generic.selection import Selection

logging.basicConfig(level=logging.INFO)
load_dotenv()


def initial_population_generator() -> List[Solution]:
    result: List[Solution] = []
    for _ in range(int(os.environ.get("POPULATION_SIZE"))):
        el = np.random.choice([0, 1], size=int(os.environ.get("STR_LEN")))
        result.append(Solution(chromosome=el))
    return result


def fitness(chromosome: np.ndarray) -> int:
    return chromosome.sum()


def stopping_criteria_fn(solution: Solution) -> bool:
    return solution.fitness == int(os.environ.get("STR_LEN"))


if __name__ == "__main__":
    params = Hyperparams(crossover_fn=Crossover.two_point,
                         initial_population_generator_fn=initial_population_generator,
                         mutation_fn=Mutation.flip_bit,
                         selection_fn=Selection.tournament,
                         fitness_fn=fitness, population_size=int(os.environ.get("POPULATION_SIZE")), elitism=5,
                         stopping_criteria_fn=stopping_criteria_fn)

    TrainingExecutor.run((params, 1))
    # TrainingExecutor.run_parallel(params)

    """
    selection_methods = [Selection.tournament, Selection.roulette, Selection.rank]
    crossover_methods = [Crossover.two_point, Crossover.single_point, Crossover.uniform]
    mutation_methods = [Mutation.flip_bit, Mutation.swap]
    population_sizes = [10, 100, 200]
    elitism_values = [1, 2, 3]

    evaluator = HyperparamEvaluator(selection_methods=selection_methods, mutation_methods=mutation_methods,
                                    crossover_methods=crossover_methods, population_sizes=population_sizes,
                                    fitness_fn=fitness, initial_population_generation_fn=initial_population_generator, elitism_values=elitism_values)

    evaluator.grid_search_parallel()
    """

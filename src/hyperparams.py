import os
from dataclasses import dataclass
from multiprocessing import Pool
from typing import List, Callable
from src.executor import TrainingExecutor
from src.model import Hyperparams


@dataclass
class HyperparamEvaluator:
    selection_methods: List[Callable]
    crossover_methods: List[Callable]
    mutation_methods: List[Callable]
    population_sizes: List[int]
    initial_population_generation_fn: Callable
    fitness_fn: Callable

    def grid_search(self):
        """
        Sequential grid search
        """
        for selection_method in self.selection_methods:
            for crossover_method in self.crossover_methods:
                for mutation_method in self.mutation_methods:
                    for population_size in self.population_sizes:
                        params = Hyperparams(mutation_fn=mutation_method, selection_fn=selection_method,
                                             crossover_fn=crossover_method,
                                             initial_population_generator_fn=self.initial_population_generation_fn,
                                             fitness_fn=self.fitness_fn, population_size=population_size)
                        TrainingExecutor.run((params, 1))

    def grid_search_parallel(self):
        """
        Parallel grid search
        """
        combinations: List[Hyperparams] = []
        for selection_method in self.selection_methods:
            for crossover_method in self.crossover_methods:
                for mutation_method in self.mutation_methods:
                    for population_size in self.population_sizes:
                        params = Hyperparams(mutation_fn=mutation_method, selection_fn=selection_method,
                                             crossover_fn=crossover_method,
                                             initial_population_generator_fn=self.initial_population_generation_fn,
                                             fitness_fn=self.fitness_fn, population_size=population_size)
                        combinations.append(params)

        with Pool(processes=int(os.environ.get("N_PROCESSES"))) as pool:
            result = pool.map_async(TrainingExecutor.run, zip(combinations, range(len(combinations))))
            result.wait()

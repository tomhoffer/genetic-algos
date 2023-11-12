import os
from dataclasses import dataclass
from multiprocessing import Pool
from typing import List, Callable, Dict, Optional, Tuple
from src.generic.executor import TrainingExecutor
from src.generic.model import Hyperparams
from sklearn.model_selection import ParameterGrid
from numpy import ndarray
import src.generic.types as types

@dataclass
class HyperparamEvaluator:
    selection_method: List[types.SelectionMethodSignature]
    crossover_method: List[types.CrossoverMethodSignature]
    mutation_method: List[types.MutationMethodSignature]
    population_size: List[int]
    initial_population_generation_fn: types.PopulationGeneratorMethodSignature
    fitness_fn: types.FitnessMethodSignature
    elitism_value: List[int]
    stopping_criteria_fn: types.StoppingCriteriaMethodSignature
    chromosome_validator_fn: types.ChromosomeValidatorMethodSignature

    def _get_hyperparam_grid(self) -> List[Dict]:
        params = self.__dict__.keys()
        values = self.__dict__.values()
        params_obj = {}

        for param, value in zip(params, values):
            if isinstance(value, List):
                params_obj[str(param)] = value
            else:
                # ParameterGrid requires single values to be wrapped inside a List
                params_obj[str(param)] = [value]

        return list(ParameterGrid(params_obj))

    def get_hyperparam_combinations(self) -> List[Hyperparams]:
        return [Hyperparams(mutation_fn=combination['mutation_method'],
                            selection_fn=combination['selection_method'],
                            crossover_fn=combination['crossover_method'],
                            initial_population_generator_fn=self.initial_population_generation_fn,
                            fitness_fn=self.fitness_fn, population_size=combination['population_size'],
                            elitism=combination['elitism_value'], stopping_criteria_fn=self.stopping_criteria_fn,
                            chromosome_validator_fn=self.chromosome_validator_fn) for combination in
                self._get_hyperparam_grid()]

    def grid_search(self, return_global_winner=False):
        """
        Sequential grid search
        """
        combinations: List[Hyperparams] = self.get_hyperparam_combinations()
        for combination in combinations:
            TrainingExecutor.run((combination, 1), return_global_winner=return_global_winner)

    def grid_search_parallel(self, return_global_winner=False):
        """
        Parallel grid search
        """
        combinations: List[Hyperparams] = self.get_hyperparam_combinations()
        with Pool(processes=int(os.environ.get("N_PROCESSES"))) as pool:
            result = pool.map_async(TrainingExecutor.run,
                                    zip(combinations, range(len(combinations)),
                                        [return_global_winner for _ in range(len(combinations))]))
            result.wait()

from dataclasses import dataclass
from typing import List
from src.generic.hyperparams import HyperparamEvaluator
from src.generic.model import Hyperparams


@dataclass
class TradingBotHyperparams(Hyperparams):
    pass


@dataclass
class TradingBotHyperparamEvaluator(HyperparamEvaluator):

    def get_hyperparam_combinations(self) -> List[TradingBotHyperparams]:
        return [TradingBotHyperparams(mutation_fn=combination['mutation_method'],
                                      selection_fn=combination['selection_method'],
                                      crossover_fn=combination['crossover_method'],
                                      initial_population_generator_fn=self.initial_population_generation_fn,
                                      fitness_fn=self.fitness_fn, population_size=combination['population_size'],
                                      elitism=combination['elitism_value'],
                                      stopping_criteria_fn=self.stopping_criteria_fn,
                                      chromosome_validator_fn=self.chromosome_validator_fn) for combination in
                self._get_hyperparam_grid()]

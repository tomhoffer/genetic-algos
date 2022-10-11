from dataclasses import dataclass
from typing import List, Callable, Any


@dataclass
class HyperparamEvaluator:
    #selection_methods: List[SelectionMethod]
    #crossover_methods: List[CrossoverMethod]
    #mutation_methods: List[MutationMethod]
    #mutation_methods: List[MutationMethod]
    population_sizes: List[int]


@dataclass
class Hyperparams:
    fitness_fn: Callable
    initial_population_generator_fn: Callable
    mutation_fn: Any  # TODO type to MutationMethod
    selection_fn: Any  # TODO type to SelectionMethod
    crossover_fn: Any  # TODO type to CrossoverMethod

from typing import Type, Callable, List, Optional, Tuple
from numpy import ndarray

from src.generic.model import Population, Solution

SelectionMethodSignature = Type[Callable[[Population], List[Solution]]]
CrossoverMethodSignature = Type[Callable[[Solution, Solution], Optional[Tuple[Solution, Solution]]]]
MutationMethodSignature = Type[Callable[[ndarray], ndarray]]
PopulationGeneratorMethodSignature = Type[Callable[[], List[Solution]]]
FitnessMethodSignature = Type[Callable[[ndarray], float]]
StoppingCriteriaMethodSignature = Type[Callable[[Solution], bool]]
ChromosomeValidatorMethodSignature = Type[Callable[[Solution], bool]]

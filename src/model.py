from dataclasses import dataclass
from typing import List, Callable, Tuple, Any


@dataclass
class Solution:
    chromosome: str = ""  # TODO make generic
    fitness: float = 0


@dataclass
class Population:
    members: List[Solution]
    fitness_fn: Callable
    mutation_fn: Any  # Cannot type to MutationMethod because of circular imports
    crossover_fn: Any  # Cannot type to CrossoverMethod because of circular imports

    def refresh_fitness(self):
        for individual in self.members:
            individual.fitness = self.fitness_fn(individual.chromosome)

    def mutate(self):
        for individual in self.members:
            individual.chromosome = self.mutation_fn(individual.chromosome)
        self.refresh_fitness()

    def perform_crossover(self):
        middle_point = len(self.members) // 2
        group1 = self.members[:middle_point]
        group2 = self.members[middle_point:]
        offsprings: List[Solution] = []
        for a, b in zip(group1, group2):
            offspring1, offspring2 = self.crossover_fn(a, b)
            offsprings.append(offspring1)
            offsprings.append(offspring2)
        self.members = offsprings
        self.refresh_fitness()

    def get_winner(self) -> Tuple[Solution, int]:
        """
        :return: Tuple [winner, fitness]
        """

        max_fitness = 0
        winner = None
        self.refresh_fitness()
        for el in self.members:
            if el.fitness > max_fitness:
                winner = el
                max_fitness = el.fitness
        return winner, max_fitness

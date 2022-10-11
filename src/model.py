import logging
from dataclasses import dataclass
from typing import List, Tuple, Callable, Any

from src.conf import CONFIG


@dataclass
class Hyperparams:
    fitness_fn: Callable
    initial_population_generator_fn: Callable
    mutation_fn: Any  # TODO type to MutationMethod
    selection_fn: Any  # TODO type to SelectionMethod
    crossover_fn: Any  # TODO type to CrossoverMethod
    population_size: int


@dataclass
class Solution:
    chromosome: str = ""  # TODO make generic
    fitness: float = 0


@dataclass
class Population(Hyperparams):
    members: List[Solution]

    def train(self, id: int) -> Tuple[Solution, bool, int]:
        """
        :param self:
        :param id: Process identifier
        :return: Solution, boolean indicating whether result was found or not, process identifier
        """

        logging.debug(f"Process started with id {id}...")

        winner = None
        success = False
        self.generate_initial_population()

        for i in range(CONFIG["MAX_ITERS"]):
            self.perform_selection()
            self.perform_crossover()
            self.mutate()
            winner, winner_fitness = self.get_winner()

            # Stopping criteria - If string contains all 1s
            if winner_fitness == CONFIG["STR_LEN"]:
                success = True
                logging.debug(f"Found result after {i} iterations in process {id}: {winner}!")
                break
        return winner, success, id

    def generate_initial_population(self):
        self.members = self.initial_population_generator_fn()
        self.refresh_fitness()

    def refresh_fitness(self):
        for individual in self.members:
            individual.fitness = self.fitness_fn(individual.chromosome)

    def mutate(self):
        for individual in self.members:
            individual.chromosome = self.mutation_fn(individual.chromosome)
        self.refresh_fitness()

    def perform_selection(self):
        self.members = self.selection_fn(self)
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

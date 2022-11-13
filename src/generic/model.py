import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Callable
import numpy as np
import wandb

from src.generic.helpers import eval_bool


@dataclass
class Hyperparams:
    fitness_fn: Callable
    initial_population_generator_fn: Callable
    mutation_fn: Callable
    selection_fn: Callable
    crossover_fn: Callable
    stopping_criteria_fn: Callable
    population_size: int
    elitism: int


@dataclass
class Solution:
    chromosome: np.ndarray
    fitness: float = 0

    def __eq__(self, other):
        return np.array_equal(self.chromosome, other.chromosome)


class PopulationBase(ABC):

    @abstractmethod
    def train(self, id: int):
        pass

    @abstractmethod
    def generate_initial_population(self):
        pass

    @abstractmethod
    def refresh_fitness(self):
        pass

    @abstractmethod
    def perform_mutation(self):
        pass

    @abstractmethod
    def perform_selection(self):
        pass

    @abstractmethod
    def perform_crossover(self):
        pass

    @abstractmethod
    def get_winner(self):
        pass


@dataclass
class Population(Hyperparams, PopulationBase):
    members: List[Solution]

    def train(self, id: int) -> Tuple[Solution, bool, int]:

        """
        :param self:
        :param id: Process identifier
        :return: Solution, boolean indicating whether result was found or not, process identifier
        """

        logging.debug("Process started with id %d...", id)

        if eval_bool(os.environ.get("ENABLE_WANDB")):
            run = wandb.init(project="genetic-algos-one-max", config={
                "crossover-method": self.crossover_fn,
                "selection-method": self.selection_fn,
                "mutation-method": self.mutation_fn,
                "population-size": self.population_size,
            }, reinit=True)

        winner = None
        success = False
        self.generate_initial_population()

        for i in range(int(os.environ.get("MAX_ITERS"))):
            self.perform_selection()
            self.perform_crossover()
            self.perform_mutation()
            winner, winner_fitness = self.get_winner()

            if eval_bool(os.environ.get("ENABLE_WANDB")):
                wandb.log({
                    "fitness": winner_fitness
                }, step=i)

            if self.stopping_criteria_fn(winner):
                success = True
                logging.info("Found result after %d iterations in process %d: %s!", i, id, winner)
                break
        if eval_bool(os.environ.get("ENABLE_WANDB")):
            run.finish()

        return winner, success, id

    def generate_initial_population(self):
        self.members = self.initial_population_generator_fn()
        self.refresh_fitness()

    def refresh_fitness(self):
        for individual in self.members:
            individual.fitness = self.fitness_fn(individual.chromosome)

    def perform_mutation(self):
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

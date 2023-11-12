import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean
from typing import List, Tuple
import numpy as np
import wandb
from tqdm import tqdm

from src.generic.config import Config
from src.tradingbot import redis_connector
from src.generic.helpers import eval_bool, hash_chromosome
import src.generic.types as types

redis_conn = redis_connector.connect()


class InvalidPopulationException(Exception):
    pass


@dataclass
class Hyperparams:
    fitness_fn: types.FitnessMethodSignature
    initial_population_generator_fn: types.PopulationGeneratorMethodSignature
    mutation_fn: types.MutationMethodSignature
    selection_fn: types.SelectionMethodSignature
    crossover_fn: types.CrossoverMethodSignature
    stopping_criteria_fn: types.StoppingCriteriaMethodSignature
    chromosome_validator_fn: types.ChromosomeValidatorMethodSignature
    population_size: int
    elitism: int

    def get_wandb_config(self):
        config = {}
        for key, val in zip(self.__dict__.keys(), self.__dict__.values()):
            config[key] = val
        return config


@dataclass
class Solution:
    chromosome: np.ndarray
    fitness: float = 0

    def __eq__(self, other):
        return np.array_equal(self.chromosome, other.chromosome)

    def serialize_to_file(self, path: str):
        np.save(path, self.chromosome)


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
class Population(PopulationBase):
    members: List[Solution]
    hyperparams: Hyperparams

    def train(self, id: int, return_global_winner=False) -> Tuple[Solution, bool, int]:

        """
        :param self:
        :param id: Process identifier
        :param wandb_config: Config object for wandb service
        :param return_global_winner: Return the best solution found along the path instead of the last one
        :return: Solution, boolean indicating whether result was found or not, process identifier
        """

        logging.debug("Process started with id %d...", id)

        if eval_bool(os.environ.get("ENABLE_WANDB")):
            run = wandb.init(project="genetic-algos", config=self.hyperparams.get_wandb_config(), reinit=True)

        winner = None
        global_winner = None
        global_winner_fitness: float = - np.inf
        success = False
        self.generate_initial_population()

        for i in tqdm(range(int(os.environ.get("MAX_ITERS"))), desc="Running epoch"):
            self.refresh_fitness()
            self.perform_selection()
            self.perform_crossover()
            self.perform_mutation()
            self.refresh_fitness()

            winner, winner_fitness = self.get_winner()
            if winner_fitness > global_winner_fitness:
                global_winner_fitness = winner_fitness
                global_winner = winner

            if eval_bool(os.environ.get("ENABLE_WANDB")):
                wandb_log = {
                    "fitness": winner_fitness,
                    "avg_fitness": self.get_average_fitness()
                }
                for j in range(len(winner.chromosome)):
                    wandb_log[str(j)] = winner.chromosome[j]
                wandb.log(wandb_log, step=i)

            if self.hyperparams.stopping_criteria_fn(winner):
                success = True
                logging.info("Found result after %d iterations in process %d: %s!", i, id, winner)
                break
        if eval_bool(os.environ.get("ENABLE_WANDB")):
            run.finish()
        self.refresh_fitness()

        if return_global_winner:
            return global_winner, success, id
        else:
            return winner, success, id

    def generate_initial_population(self):
        self.members = self.hyperparams.initial_population_generator_fn()
        max_attempts = 50
        attempt = 1
        while attempt <= max_attempts and not self.is_valid_population():
            self.members = self.hyperparams.initial_population_generator_fn()
            attempt += 1

        if attempt > max_attempts:
            logging.error("Unable to generate initial population within max number of attempts, quitting...")
            raise InvalidPopulationException

    def refresh_fitness(self):
        if Config.get_value('USE_REDIS_FITNESS_CACHE'):
            chromosomes_hashed: List[str] = [hash_chromosome(member.chromosome) for member in self.members]
            cached_results = redis_conn.mget(chromosomes_hashed)

            for individual, cached_result in zip(self.members, cached_results):
                if cached_result is None:
                    individual.fitness = self.hyperparams.fitness_fn(individual.chromosome)
                    redis_conn.set(hash_chromosome(individual.chromosome), str(individual.fitness))
                else:
                    individual.fitness = float(cached_result)

        else:
            for individual in self.members:
                individual.fitness = self.hyperparams.fitness_fn(individual.chromosome)

    def perform_mutation(self):
        for individual in self.members:
            individual.chromosome = self.hyperparams.mutation_fn(individual.chromosome)

    def perform_selection(self):
        self.members = self.hyperparams.selection_fn(self)

    def perform_crossover(self):
        middle_point = len(self.members) // 2
        group1 = self.members[:middle_point]
        group2 = self.members[middle_point:]
        offsprings: List[Solution] = []
        for a, b in zip(group1, group2):
            offspring1, offspring2 = self.hyperparams.crossover_fn(a, b)
            offsprings.append(offspring1)
            offsprings.append(offspring2)
        self.members = offsprings

    def get_winner(self) -> Tuple[Solution, int]:
        """
        :return: Tuple [winner, fitness]
        """

        max_fitness = np.NINF
        winner = self.members[0]
        for el in self.members:
            if el.fitness > max_fitness:
                winner = el
                max_fitness = el.fitness
        return winner, max_fitness

    def get_average_fitness(self) -> float:
        """
        :return: average fitness in the population
        """
        return mean([el.fitness for el in self.members])

    def is_valid_population(self) -> bool:
        """
            :return: True if all population members are valid
        """
        for member in self.members:
            if not self.hyperparams.chromosome_validator_fn(member):
                return False
        return True

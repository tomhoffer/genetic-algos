import copy
import logging
import random
from typing import List

from src.decorators import validate_population_length
from src.model import Population, Solution


class Selection:

    @staticmethod
    @validate_population_length
    def tournament(population: Population, tournament_size=3) -> List[Solution]:

        if all(el.fitness == 0 for el in population.members):
            logging.debug(
                "Population chosen for selection has all fitness values equal to 0. Returning original population...")
            return copy.deepcopy(population.members)

        offspring_population = []

        for _ in range(len(population.members)):
            picked: List[Solution] = random.choices(population.members, k=tournament_size)
            max_fitness = 0
            winner = None
            for el in picked:
                if el.fitness > max_fitness:
                    winner = el
                    max_fitness = el.fitness
            offspring_population.append(winner)

        logging.debug("Returning population after selection: %s", offspring_population)
        return offspring_population

    @staticmethod
    @validate_population_length
    def roulette(population: Population) -> List[Solution]:
        logging.debug("Performing roulette wheel selection from population: %s", population.members)
        sum_fitness = sum(el.fitness for el in population.members)

        if sum_fitness == 0:
            logging.debug("Roulette selection was not possible, all fitness are zero. Returning parent population...")
            return copy.deepcopy(population.members)

        selection_probs = [el.fitness / sum_fitness for el in population.members]
        logging.debug("Roulette weights for population are: %s", selection_probs)

        offspring_population = random.choices(population.members, weights=selection_probs, k=len(population.members))
        logging.debug("Returning population after selection: %s", offspring_population)
        return offspring_population

    @staticmethod
    @validate_population_length
    def rank(population: Population) -> List[Solution]:
        logging.debug("Performing rank-based selection from population: %s", population.members)

        if all(el.fitness == 0 for el in population.members):
            logging.debug(
                "Rank-based selection was not possible, all fitness are zero. Returning parent population...")
            return copy.deepcopy(population.members)

        members_ordered_by_fitness = sorted(population.members, key=lambda el: el.fitness)
        logging.debug("Population members ordered by fitness values: %s", members_ordered_by_fitness)

        sum_rank = len(members_ordered_by_fitness) * (len(members_ordered_by_fitness) + 1) // 2
        logging.debug("Rank sum is: %d", sum_rank)

        selection_probs = [(idx + 1) / sum_rank for idx, el in enumerate(members_ordered_by_fitness)]
        logging.debug("Rank weights for population are: %s", selection_probs)

        offspring_population = random.choices(members_ordered_by_fitness, weights=selection_probs,
                                              k=len(members_ordered_by_fitness))
        logging.debug("Returning population after selection: %s", offspring_population)
        return offspring_population

    # TODO stochastic universal sampling selection

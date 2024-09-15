import copy
import logging
import random
from typing import List
import numpy as np
from src.generic.config import Config
from src.generic.decorators import validate_population_length
from src.generic.model import Solution


class Selection:

    @staticmethod
    @validate_population_length
    def tournament(population: List[Solution], tournament_size=3) -> List[Solution]:

        if all(el.fitness == -np.inf for el in population):
            logging.debug(
                "Population chosen for selection has all fitness values equal to negative inf. Returning original population...")
            return copy.deepcopy(population)

        offspring_population = []
        members_to_select = len(population)  # TODO re-use code
        num_elites: int = Config.get_value('ELITISM')
        if num_elites > 0:
            members_to_select = members_to_select - num_elites
            elites: List[Solution] = _find_best(population, num_elites)
            offspring_population.extend(elites)

            for el in offspring_population:
                population.remove(el)

        for _ in range(members_to_select):
            picked: List[Solution] = random.choices(population, k=tournament_size)
            max_fitness = -np.inf
            winner = picked[0]
            for el in picked:
                if el.fitness > max_fitness:
                    winner = el
                    max_fitness = el.fitness
            offspring_population.append(winner)

        logging.debug("Returning population after selection: %s", offspring_population)
        return offspring_population

    @staticmethod
    @validate_population_length
    def roulette(population: List[Solution]) -> List[Solution]:
        logging.debug("Performing roulette wheel selection from population: %s", population)
        sum_fitness = sum(el.fitness for el in population)

        if sum_fitness == -np.inf:
            logging.error(
                "Roulette selection was not possible, all fitness are negative inf. Returning parent population...")
            return copy.deepcopy(population)

        offspring_population = []
        members_to_select = len(population)
        num_elites: int = Config.get_value('ELITISM')
        if num_elites > 0:
            members_to_select = members_to_select - num_elites
            elites: List[Solution] = _find_best(population, num_elites)
            offspring_population.extend(elites)

            for el in offspring_population:
                population.remove(el)

        try:
            selection_probs = [el.fitness / sum_fitness for el in population]
            logging.debug("Roulette weights for population are: %s", selection_probs)
        except ZeroDivisionError:
            logging.error("Sum of fitness equal to 0, returning parent population...")
            return copy.deepcopy(population)

        offspring_population.extend(random.choices(population, weights=selection_probs, k=members_to_select))
        logging.debug("Returning population after selection: %s", offspring_population)
        return offspring_population

    @staticmethod
    @validate_population_length
    def rank(population: List[Solution]) -> List[Solution]:
        logging.debug("Performing rank-based selection from population: %s", population)

        if all(el.fitness == -np.inf for el in population):
            logging.debug(
                "Rank-based selection was not possible, all fitness are negative inf. Returning parent population...")
            return copy.deepcopy(population)

        offspring_population = []
        members_to_select = len(population)
        num_elites: int = Config.get_value('ELITISM')
        if num_elites > 0:
            members_to_select = members_to_select - num_elites
            elites: List[Solution] = _find_best(population, num_elites)
            offspring_population.extend(elites)

            for el in offspring_population:
                population.remove(el)

        members_ordered_by_fitness = sorted(population, key=lambda el: el.fitness)
        logging.debug("Population members ordered by fitness values: %s", members_ordered_by_fitness)

        sum_rank = len(members_ordered_by_fitness) * (len(members_ordered_by_fitness) + 1) // 2
        logging.debug("Rank sum is: %d", sum_rank)

        selection_probs = [(idx + 1) / sum_rank for idx, el in enumerate(members_ordered_by_fitness)]
        logging.debug("Rank weights for population are: %s", selection_probs)

        offspring_population.extend(random.choices(members_ordered_by_fitness, weights=selection_probs,
                                                   k=members_to_select))
        logging.debug("Returning population after selection: %s", offspring_population)
        return offspring_population

    # TODO stochastic universal sampling selection


def _find_best(population: List[Solution], n: int) -> List[Solution]:
    """
    Returns n best individuals from given population
    :param population: population
    :param n: number of best individuals to return
    :return: best n individuals
    """
    return sorted(population, key=lambda x: x.fitness, reverse=True)[:n]

import copy
import logging
import random
from typing import List

from src.model import Population, Solution


def _check_population_length(population: Population):
    if len(population.members) < 1:
        raise ValueError(f"Population size lower than 1! Actual: {len(population.members)}")


def select_tournament(population: Population, tournament_size=3) -> List[Solution]:
    _check_population_length(population)

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

    logging.debug(f"Returning population after selection: {offspring_population}")
    return offspring_population


def select_roulette(population: Population) -> List[Solution]:
    _check_population_length(population)
    logging.debug(f"Performing roulette wheel selection from population: {population.members}")
    sum_fitness = sum(el.fitness for el in population.members)

    if sum_fitness == 0:
        logging.debug(f"Roulette selection was not possible, all fitness are zero. Returning parent population...")
        return copy.deepcopy(population.members)

    selection_probs = [el.fitness / sum_fitness for el in population.members]
    logging.debug(f"Roulette weights for population are: {selection_probs}")

    offspring_population = random.choices(population.members, weights=selection_probs, k=len(population.members))
    logging.debug(f"Returning population after selection: {offspring_population}")
    return offspring_population

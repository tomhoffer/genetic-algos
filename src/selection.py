import copy
import logging
import random
from typing import List

from src.model import Population, Solution


def select_tournament(population: Population, tournament_size=3) -> Population:
    if len(population.members) < 1:
        raise ValueError(f"Population size lower than 1! Actual: {len(population.members)}")

    if all(el.fitness == 0 for el in population.members):
        logging.debug(
            "Population chosen for selection has all fitness values equal to 0. Returning original population...")
        return copy.deepcopy(population)

    candidates = copy.deepcopy(population)
    offspring_population = Population(members=[], fitness_fn=population.fitness_fn, mutation_fn=population.mutation_fn,
                                      crossover_fn=population.crossover_fn)

    for _ in range(len(population.members)):
        picked: List[Solution] = random.choices(candidates.members, k=tournament_size)
        max_fitness = 0
        winner = None
        for el in picked:
            if el.fitness > max_fitness:
                winner = el
                max_fitness = el.fitness
        offspring_population.members.append(winner)

    logging.debug(f"Returning population after selection: {offspring_population}")
    return offspring_population

import copy
import logging
import random
from typing import List

from src.model import Population, Solution


def select_tournament(population: Population) -> Population:
    TOURNAMENT_SIZE = 3
    candidates = copy.deepcopy(population)
    offspring_population = Population(members=[], fitness_fn=population.fitness_fn, mutation_fn=population.mutation_fn,
                                      crossover_fn=population.crossover_fn)

    for _ in range(len(population.members)):
        picked: List[Solution] = random.choices(candidates.members, k=TOURNAMENT_SIZE)
        max_fitness = 0
        winner = None
        for el in picked:
            if el.fitness > max_fitness:
                winner = el
                max_fitness = el.fitness
        offspring_population.members.append(winner)

    logging.debug(f"Returning population after selection: {offspring_population}")
    return offspring_population

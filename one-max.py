import logging
import random
from typing import List

from src.crossover import crossover_two_point
from src.model import Population, Solution
from src.mutation import mutate
from src.selection import select_tournament

POPULATION_SIZE = 50
STR_LEN = 100
MAX_ITERS = 1000

logging.basicConfig(level=logging.DEBUG)


def generate_initial_population(size: int = POPULATION_SIZE) -> Population:
    result: List[Solution] = []
    for _ in range(size):
        el = ""
        for _ in range(STR_LEN):
            el = el + str(random.randint(0, 1))
        result.append(Solution(chromosome=el))
    p = Population(members=result, fitness_fn=fitness, mutation_fn=mutate, crossover_fn=crossover_two_point)
    p.refresh_fitness()
    return p


def fitness(chromosome: str) -> int:
    sum = 0
    for bit in chromosome:
        sum += int(bit)
    return sum


if __name__ == "__main__":

    winner: str = ""
    winner_fitness: int = 0
    population = generate_initial_population(10)

    for i in range(MAX_ITERS):
        population = select_tournament(population)
        population.perform_crossover()
        population.mutate()
        winner, winner_fitness = population.get_winner()

        # If string contains all 1s
        if winner_fitness == STR_LEN:
            logging.info(f"Found result: {winner}!")
            exit(0)

    logging.info(f"No solution found within {MAX_ITERS} iterations. Winner with fitness {winner_fitness} was: {winner}")

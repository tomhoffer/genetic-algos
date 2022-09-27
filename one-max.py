import copy
import logging
import random
from typing import Tuple, Optional, List

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


# TODO isolate
def crossover_single_point(parent1: Solution, parent2: Solution) -> Optional[Tuple[Solution, Solution]]:
    if len(parent1.chromosome) != len(parent2.chromosome):
        logging.error(f"Gene length does not match! Parents: {parent1}, {parent2}")
        return None
    crossover_pos = random.randint(0, len(parent1.chromosome) - 1)

    offspring_chromosome1 = parent1.chromosome[:crossover_pos] + parent2.chromosome[crossover_pos:]
    offspring_chromosome2 = parent2.chromosome[:crossover_pos] + parent1.chromosome[crossover_pos:]
    logging.debug(
        f"Performing crossover operation over position {crossover_pos} between parents: "
        f"{parent1.chromosome}, {parent2.chromosome}... "
        f"Results: {offspring_chromosome1}, {offspring_chromosome2}"
    )
    return Solution(chromosome=offspring_chromosome1), Solution(chromosome=offspring_chromosome2)


# TODO isolate
def crossover_two_point(a: Solution, b: Solution) -> Optional[Tuple[Solution, Solution]]:
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    logging.debug(f"Performing 2-point crossover between parents: {a.chromosome}, {b.chromosome}... ")

    for i in range(2):
        a, b = crossover_single_point(a, b)
    logging.debug(f"Crossover done, results: {a}, {b}... ")
    return a, b


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

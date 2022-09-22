import copy
import logging
import random
from typing import List, Tuple, Optional

POPULATION_SIZE = 500
STR_LEN = 50
P_MUTATION = 0.02
MAX_ITERS = 100

logging.basicConfig(level=logging.DEBUG)


def generate_initial_population(size: int = POPULATION_SIZE) -> List[str]:
    result = []
    for _ in range(size):
        el = ""
        for _ in range(STR_LEN):
            el = el + str(random.randint(0, 1))
        result.append(el)
    return result


def fitness(element: str) -> int:
    sum = 0
    for bit in element:
        sum += int(bit)
    return sum


def select_tournament(population: List[str]) -> List[str]:
    TOURNAMENT_SIZE = 3
    candidates = copy.deepcopy(population)
    offspring_population: List[str] = []

    for _ in range(len(population)):
        picked: List[str] = random.choices(candidates, k=TOURNAMENT_SIZE)
        max_fitness = 0
        winner = None
        for el in picked:
            if fitness(el) > max_fitness:
                winner = el
                max_fitness = fitness(el)
        offspring_population.append(winner)

    logging.debug(f"Returning population after selection: {offspring_population}")
    return offspring_population


def crossover_single_point(parent1: str, parent2: str) -> Optional[Tuple[str, str]]:
    if len(parent1) != len(parent2):
        logging.error(f"Gene length does not match! Parents: {parent1}, {parent2}")
        return None
    crossover_pos = random.randint(0, len(parent1) - 1)
    offspring1 = parent1[:crossover_pos] + parent2[crossover_pos:]
    offspring2 = parent2[:crossover_pos] + parent1[crossover_pos:]
    logging.debug(
        f"Performing crossover operation over position {crossover_pos} between parents: {parent1}, {parent2}... "
        f"Results: {offspring1}, {offspring2}"
    )
    return offspring1, offspring2


def crossover_two_point(a: str, b: str) -> Optional[Tuple[str, str]]:
    a = copy.deepcopy(a)
    b = copy.deepcopy(b)
    logging.debug(f"Performing 2-point crossover between parents: {a}, {b}... ")

    for i in range(2):
        a, b = crossover_single_point(a, b)
    logging.debug(f"Crossover done, results: {a}, {b}... ")
    return a, b


def perform_crossover(population: List[str]) -> List[str]:
    middle_point = len(population) // 2
    group1 = population[:middle_point]
    group2 = population[middle_point:]
    result: List[str] = []
    for a, b in zip(group1, group2):
        offspring1, offspring2 = crossover_two_point(a, b)
        result.append(offspring1)
        result.append(offspring2)
    return result


def mutate(sequence: str) -> str:
    sequence = copy.deepcopy(sequence)
    result = ""
    for gene in sequence:
        if random.random() < P_MUTATION:
            # flip the bit
            result += str(1 - int(gene))
            logging.debug(f"Mutation probability hit! Mutating gene: {sequence}...")
        else:
            result += gene
    return result


def perform_mutation(population: List[str]) -> List[str]:
    return [mutate(el) for el in population]


def evaluate_winner(population: List[str]) -> Tuple[str, int]:
    """
    :param population: Population for evaluation
    :return: Tuple [winner, fitness]
    """

    max_fitness = 0
    winner = None
    for el in population:
        if fitness(el) > max_fitness:
            winner = el
            max_fitness = fitness(el)
    return winner, max_fitness


if __name__ == "__main__":

    winner: str = ""
    winner_fitness: int = 0
    population = generate_initial_population(10)

    for i in range(MAX_ITERS):
        population = select_tournament(population)
        population = perform_crossover(population)
        population = perform_mutation(population)

        winner, winner_fitness = evaluate_winner(population)

        # If string contains all 1s
        if winner_fitness == STR_LEN:
            logging.info(f"Found result: {winner}!")
            exit(0)

    logging.info(f"No solution found within {MAX_ITERS} iterations. Winner with fitness {winner_fitness} was: {winner}")

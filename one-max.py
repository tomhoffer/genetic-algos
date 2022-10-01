import logging
import random
from multiprocessing import Pool
from typing import List, Tuple

from src.enums import CrossoverMethod, MutationMethod, SelectionMethod
from src.model import Population, Solution
from src.selection import select_tournament

POPULATION_SIZE = 10
STR_LEN = 100
MAX_ITERS = 100
N_PROCESSES = 4
TIMEOUT_SECONDS = 120

logging.basicConfig(level=logging.DEBUG)


def generate_initial_population(size: int = POPULATION_SIZE) -> Population:
    result: List[Solution] = []
    for _ in range(size):
        el = ""
        for _ in range(STR_LEN):
            el = el + str(random.randint(0, 1))
        result.append(Solution(chromosome=el))
    p = Population(members=result, fitness_fn=fitness, mutation_fn=MutationMethod.FLIP_BIT,
                   crossover_fn=CrossoverMethod.UNIFORM, selection_fn=SelectionMethod.ROULETTE)
    p.refresh_fitness()
    return p


def fitness(chromosome: str) -> int:
    return sum(int(x) for x in chromosome)


def train(id: int) -> Tuple[Solution, bool, int]:
    """
    :param id: Process identifier
    :return: Solution, boolean indicating whether result was found or not, process identifier
    """

    logging.debug(f"Process started with id {id}...")

    winner = None
    success = False
    population = generate_initial_population()

    for i in range(MAX_ITERS):
        population.perform_selection()
        population.perform_crossover()
        population.mutate()
        winner, winner_fitness = population.get_winner()

        # Stopping criteria - If string contains all 1s
        if winner_fitness == STR_LEN:
            success = True
            logging.debug(f"Found result after {i} iterations in process {id}: {winner}!")
            break
    return winner, success, id


if __name__ == "__main__":
    with Pool(processes=N_PROCESSES) as pool:
        it = pool.imap_unordered(train, range(N_PROCESSES))

        winner = Solution(chromosome="", fitness=0)
        winner_process_id: int = 0

        while True:  # TODO handle timeout

            try:
                # Get the first result, blocking
                result, success, process_id = next(it)

                if success:
                    winner = result
                    winner_process_id = process_id
                    logging.info(f"Solution found in process {process_id}! {result}")
                    logging.debug("Killing other processes...")
                    pool.close()
                    break
                else:
                    if result.fitness > winner.fitness:
                        winner = result
                        winner_process_id = process_id

            # If there is no more values in iterator
            except StopIteration:
                logging.info(
                    f"No solution found within {MAX_ITERS} iterations. Winner with fitness {winner.fitness} from process {winner_process_id} was: {winner.chromosome}")
                break

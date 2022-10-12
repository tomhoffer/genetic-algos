import logging
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Tuple

from src.conf import CONFIG
from src.model import Population, Solution, Hyperparams


@dataclass
class TrainingExecutor:

    @staticmethod
    def run(args: Tuple[Hyperparams, int]) -> Tuple[Solution, bool, int]:
        params: Hyperparams = args[0]
        process_id: int = args[1]
        logging.info(f"Running training with parameters: {params}")

        p = Population(fitness_fn=params.fitness_fn,
                       initial_population_generator_fn=params.initial_population_generator_fn,
                       selection_fn=params.selection_fn, mutation_fn=params.mutation_fn,
                       crossover_fn=params.crossover_fn, members=[], population_size=params.population_size)

        return p.train(id=process_id)

    @staticmethod
    def run_parallel(params: Hyperparams):
        logging.info(f"Running parallel training with parameters: {params}")
        with Pool(processes=CONFIG["N_PROCESSES"]) as pool:

            it = pool.imap_unordered(TrainingExecutor.run,
                                     zip([params for _ in range(CONFIG["N_PROCESSES"])], range(CONFIG["N_PROCESSES"])))

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
                        logging.info("Killing other processes...")
                        pool.close()
                        break
                    else:
                        if result.fitness > winner.fitness:
                            winner = result
                            winner_process_id = process_id

                # If there is no more values in iterator
                except StopIteration:
                    logging.info(
                        f"No solution found within {CONFIG['MAX_ITERS']} iterations. Winner with fitness {winner.fitness} from process {winner_process_id} was: {winner.chromosome}")
                    break

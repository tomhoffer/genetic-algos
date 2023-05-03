import logging
import os
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Tuple

import numpy as np

from src.generic.model import Population, Solution, Hyperparams


@dataclass
class TrainingExecutor:

    @staticmethod
    def run(args: Tuple[Hyperparams, int], return_global_winner=False) -> Tuple[Solution, bool, int]:
        params: Hyperparams = args[0]
        process_id: int = args[1]
        logging.info(f"Running training with parameters: {params}")
        return Population(members=[], hyperparams=params).train(id=process_id,
                                                                return_global_winner=return_global_winner)

    @staticmethod
    def run_parallel(params: Hyperparams, return_global_winner=False) -> Tuple[Solution, bool, int]:
        logging.info(f"Running parallel training with parameters: {params}")
        with Pool(processes=int(os.environ.get("N_PROCESSES"))) as pool:

            n_processes = int(os.environ.get("N_PROCESSES"))
            it = pool.imap_unordered(TrainingExecutor.run,
                                     zip([params for _ in range(n_processes)],
                                         range(n_processes), [return_global_winner for _ in range(n_processes)]))

            winner = Solution(chromosome=np.asarray([]), fitness=0)
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
                        f"No solution found within {os.environ.get('MAX_ITERS')} iterations. Winner with fitness {winner.fitness} from process {winner_process_id} was: {winner.chromosome}")
                    break
        return winner, False, winner_process_id

import logging
import os
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Tuple, List

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
    def run_parallel(params: Hyperparams, return_global_winner=False, n_runs=None, return_all_winners=False) -> Tuple[
        Solution | List[Solution], bool, int]:
        """
        :param params: Hyperparameters to use
        :param return_global_winner: If true, global winner is returned instead of the best individual in the last epoch
        :param n_runs: Total number of runs. Example: n_runs=10, N_PROCESSES=5 -> Total 10 runs will be executed across
        5 parallel processes
        :param return_all_winners: If true and no solution is found, winners from all runs are returned instead of the
        single best winner
        :return:
        """

        logging.info(f"Running parallel training with parameters: {params}")
        all_winners: List[Solution] = []
        n_processes = int(os.environ.get("N_PROCESSES"))
        if not n_runs:
            n_runs = int(os.environ.get("N_PROCESSES"))
        with Pool(processes=n_processes) as pool:

            it = pool.imap_unordered(TrainingExecutor.run,
                                     zip([params for _ in range(n_runs)],
                                         range(n_runs), [return_global_winner for _ in range(n_runs)]))

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
                        if return_all_winners:
                            all_winners.append(result)

                # If there is no more values in iterator
                except StopIteration:
                    logging.info(
                        f"No solution found within {os.environ.get('MAX_ITERS')} iterations. Winner with fitness {winner.fitness} from process {winner_process_id} was: {winner.chromosome}")
                    break

        if return_all_winners:
            return all_winners, False, 0

        return winner, False, winner_process_id

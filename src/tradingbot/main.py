import numpy as np

from src.generic.crossover import Crossover
from src.generic.executor import TrainingExecutor
from src.generic.selection import Selection
from src.tradingbot.backtest import BacktestExecutor
from src.tradingbot.config import Config
from src.tradingbot.hyperparams import TradingBotHyperparams
from src.tradingbot.tradingbot import initial_population_generator, mutate_uniform, fitness, stopping_criteria_fn, \
    chromosome_validator_fn, timestamp_to_str, TradingbotSolution, get_trading_strategy_method_names

if __name__ == "__main__":

    params = TradingBotHyperparams(crossover_fn=Crossover.two_point,
                                   initial_population_generator_fn=initial_population_generator,
                                   mutation_fn=mutate_uniform,
                                   selection_fn=Selection.tournament,
                                   fitness_fn=fitness, population_size=Config.get_value("POPULATION_SIZE"), elitism=1,
                                   stopping_criteria_fn=stopping_criteria_fn,
                                   chromosome_validator_fn=chromosome_validator_fn)

    print(
        f"Training on period: {timestamp_to_str(Config.get_value('START_TIMESTAMP'))} - {timestamp_to_str(Config.get_value('END_TIMESTAMP'))}")

    winners, _, _ = TrainingExecutor.run_parallel(params, return_global_winner=True, return_all_winners=True, n_runs=80)

    winner = None
    winner_profit_ratio = -np.inf

    for w in winners:
        backtest_executor = BacktestExecutor()
        backtest_executor.backtest(w, start_date=timestamp_to_str(Config.get_value('START_TIMESTAMP')),
                                   end_date=timestamp_to_str(Config.get_value('END_TIMESTAMP')))
        profit_ratio = backtest_executor.compute_profit_ratio()

        if profit_ratio > winner_profit_ratio:
            winner = TradingbotSolution(chromosome=w.chromosome)
            winner.fitness = w.fitness
            winner_profit_ratio = profit_ratio
    print(
        f"Found winner with weights {[el for el in zip(get_trading_strategy_method_names(), winner.chromosome)]}, resulting account balance {winner.fitness} and payoff ratio {winner_profit_ratio}")

    backtesting_periods = [("2022-01-01", "2023-01-01"), ("2023-01-01", "2024-01-01"), ("2024-01-01", "2024-08-28")]

    for backtesting_period in backtesting_periods:
        result: float = BacktestExecutor().backtest(winner, plot=True, start_date=backtesting_period[0],
                                                    end_date=backtesting_period[1], print_results=True)

    # backtest_winner.serialize_to_file('storage/weights.csv')

    # winners, _, _ = TrainingExecutor.run((params, 1), return_global_winner=True)

    """
    # selection_methods = [Selection.tournament, Selection.roulette, Selection.rank]
    selection_methods = [Selection.tournament]
    # crossover_methods = [Crossover.two_point, Crossover.single_point, Crossover.uniform]
    crossover_methods = [Crossover.two_point]
    mutation_methods = [mutate_uniform]
    population_sizes = [100, 200, 500, 750]
    elitism_values = [1, 5, 10, 50]

    evaluator = TradingBotHyperparamEvaluator(selection_method=selection_methods, mutation_method=mutation_methods,
                                              crossover_method=crossover_methods, population_size=population_sizes,
                                              fitness_fn=fitness,
                                              initial_population_generation_fn=initial_population_generator,
                                              elitism_value=elitism_values, stopping_criteria_fn=stopping_criteria_fn,
                                              chromosome_validator_fn=chromosome_validator_fn)

    evaluator.grid_search_parallel(return_global_winner=True)
    """

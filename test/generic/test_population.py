import copy

import numpy as np
import pytest

from src.generic.model import Solution, InvalidPopulationException, Population, Hyperparams
from conftest import mockenv


def test_get_winner(population_with_growing_fitness, configurable_hyperparams):
    population = Population(members=population_with_growing_fitness, hyperparams=configurable_hyperparams)
    winner, fitness = population.get_winner()

    assert winner.fitness == fitness
    assert fitness == max(member.fitness for member in population.members)


def test_perform_selection(population_with_growing_fitness, configurable_hyperparams):
    population = Population(members=population_with_growing_fitness, hyperparams=configurable_hyperparams)
    old_members = copy.deepcopy(population.members)
    population.perform_selection()
    assert not population.members == old_members


def test_perform_mutation(population_with_growing_fitness, configurable_hyperparams):
    population = Population(members=population_with_growing_fitness, hyperparams=configurable_hyperparams)
    old_members = copy.deepcopy(population.members)
    population.perform_mutation()
    assert not population.members == old_members


@pytest.mark.crossover_fn(lambda a, b: (a, b))
def test_perform_crossover(population_with_growing_fitness, configurable_hyperparams):
    population = Population(members=population_with_growing_fitness, hyperparams=configurable_hyperparams)
    old_members = copy.deepcopy(population.members)
    population.perform_crossover()
    assert not population.members == old_members


@pytest.mark.initial_population_generator_fn(lambda: [1, 2, 3, 4, 5])  # Dummy population generator
def test_generate_initial_population(empty_population, configurable_hyperparams, mocker):
    population = Population(members=empty_population, hyperparams=configurable_hyperparams)
    mocked_is_valid_population = mocker.patch('src.generic.model.Population.is_valid_population')
    old_members = copy.deepcopy(population.members)

    # Re-generate population again
    population.generate_initial_population()

    mocked_is_valid_population.assert_called_once()
    assert not population.members == old_members


def test_generate_initial_population_invalid(empty_population, configurable_hyperparams, mocker):
    # Population is always invalid, exception is Raised
    population = Population(members=empty_population, hyperparams=configurable_hyperparams)
    mocked_refresh_fitness = mocker.patch('src.generic.model.Population.refresh_fitness')
    mocked_is_valid_population = mocker.patch('src.generic.model.Population.is_valid_population')
    mocked_is_valid_population.return_value = False

    with pytest.raises(InvalidPopulationException):
        population.generate_initial_population()

    assert mocked_is_valid_population.call_count == 50
    assert mocked_refresh_fitness.call_count == 0


@pytest.mark.stopping_criteria_fn(lambda a: a.fitness == np.inf)
@pytest.mark.fitness_fn(lambda a: 1)
@mockenv(MAX_ITERS="3", ENABLE_WANDB="True", STR_LEN="3", USE_REDIS_FITNESS_CACHE="false")
def test_train_not_successful(population_with_growing_fitness, configurable_hyperparams, mocker):
    # Test solution not being found within max iterations
    population = Population(members=population_with_growing_fitness, hyperparams=configurable_hyperparams)
    max_iters = 3
    id = 1
    mocked_generate_initial_population = mocker.patch('src.generic.model.Population.generate_initial_population')
    mocked_perform_selection = mocker.patch('src.generic.model.Population.perform_selection')
    mocked_perform_crossover = mocker.patch('src.generic.model.Population.perform_crossover')
    mocked_perform_mutation = mocker.patch('src.generic.model.Population.perform_mutation')
    mocked_get_winner = mocker.patch('src.generic.model.Population.get_winner')
    mocked_get_winner.return_value = (Solution(chromosome=np.asarray([1, 0, 1]), fitness=0), 0)
    mocked_wandb_init = mocker.patch('wandb.init')
    mocked_wandb_log = mocker.patch('wandb.log')

    winner, success, process_id = population.train(id=id)

    assert mocked_generate_initial_population.call_count == 1
    assert mocked_wandb_init.call_count == 1
    assert mocked_wandb_log.call_count == max_iters
    assert mocked_perform_selection.call_count == max_iters
    assert mocked_perform_crossover.call_count == max_iters
    assert mocked_perform_mutation.call_count == max_iters
    assert mocked_get_winner.call_count == max_iters
    assert mocked_get_winner.call_count == max_iters

    assert winner == Solution(chromosome=np.asarray([1, 0, 1]), fitness=0)
    assert success is False
    assert process_id == id


@pytest.mark.stopping_criteria_fn(lambda a: a.fitness == 1)
@pytest.mark.fitness_fn(lambda a: 1)
@mockenv(MAX_ITERS="3", ENABLE_WANDB="True", STR_LEN="3", USE_REDIS_FITNESS_CACHE="false")
def test_train_successful(population_with_growing_fitness, configurable_hyperparams, mocker):
    # Test solution being found after first iteration
    population = Population(members=population_with_growing_fitness, hyperparams=configurable_hyperparams)
    id = 1
    mocked_generate_initial_population = mocker.patch('src.generic.model.Population.generate_initial_population')
    mocked_perform_selection = mocker.patch('src.generic.model.Population.perform_selection')
    mocked_perform_crossover = mocker.patch('src.generic.model.Population.perform_crossover')
    mocked_perform_mutation = mocker.patch('src.generic.model.Population.perform_mutation')
    mocked_get_winner = mocker.patch('src.generic.model.Population.get_winner')
    mocked_get_winner.return_value = (Solution(chromosome=np.asarray([1, 1, 1]), fitness=1), 3)
    mocked_wandb_init = mocker.patch('wandb.init')
    mocked_wandb_log = mocker.patch('wandb.log')

    winner, success, process_id = population.train(id=id)

    assert mocked_generate_initial_population.call_count == 1
    assert mocked_wandb_init.call_count == 1
    assert mocked_wandb_log.call_count == 1
    assert mocked_perform_selection.call_count == 1
    assert mocked_perform_crossover.call_count == 1
    assert mocked_perform_mutation.call_count == 1
    assert mocked_get_winner.call_count == 1
    assert mocked_get_winner.call_count == 1

    assert winner == Solution(chromosome=np.asarray([1, 1, 1]), fitness=3)
    assert success is True
    assert process_id == id


def _stopping_criteria_fn(winner: Solution):
    return winner.fitness == 3

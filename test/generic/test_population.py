import copy

import numpy as np
import pytest

from src.generic.model import Solution, InvalidPopulationException
from conftest import mockenv


def test_get_winner(configurable_population, mocker):
    population = configurable_population(elitism=0)
    mocked_refresh_fitness = mocker.patch('src.generic.model.Population.refresh_fitness')
    winner, fitness = population.get_winner()

    mocked_refresh_fitness.assert_called_once()
    assert winner.fitness == fitness
    assert fitness == max(member.fitness for member in population.members)


def test_perform_selection(configurable_population, mocker):
    population = configurable_population(elitism=0)
    mocked_refresh_fitness = mocker.patch('src.generic.model.Population.refresh_fitness')
    old_members = copy.deepcopy(population.members)
    population.perform_selection()
    mocked_refresh_fitness.assert_called_once()
    assert not population.members == old_members


def test_perform_mutation(configurable_population, mocker):
    population = configurable_population(elitism=0)
    mocked_refresh_fitness = mocker.patch('src.generic.model.Population.refresh_fitness')
    old_members = copy.deepcopy(population.members)
    population.perform_mutation()
    mocked_refresh_fitness.assert_called_once()
    assert not population.members == old_members


def test_perform_crossover(configurable_population, mocker):
    population = configurable_population(elitism=0)
    mocked_refresh_fitness = mocker.patch('src.generic.model.Population.refresh_fitness')
    old_members = copy.deepcopy(population.members)
    population.perform_crossover()
    mocked_refresh_fitness.assert_called_once()
    assert not population.members == old_members


def test_generate_initial_population(empty_population, mocker):
    mocked_refresh_fitness = mocker.patch('src.generic.model.Population.refresh_fitness')
    mocked_is_valid_population = mocker.patch('src.generic.model.Population.is_valid_population')
    old_members = copy.deepcopy(empty_population.members)

    # Re-generate population again
    empty_population.generate_initial_population()

    mocked_refresh_fitness.assert_called_once()
    mocked_is_valid_population.assert_called_once()
    assert not empty_population.members == old_members


def test_generate_initial_population_invalid(empty_population, mocker):
    # Population is always invalid, exception is Raised
    mocked_refresh_fitness = mocker.patch('src.generic.model.Population.refresh_fitness')
    mocked_is_valid_population = mocker.patch('src.generic.model.Population.is_valid_population')
    mocked_is_valid_population.return_value = False

    with pytest.raises(InvalidPopulationException):
        empty_population.generate_initial_population()

    assert mocked_is_valid_population.call_count == 50
    assert mocked_refresh_fitness.call_count == 0


@mockenv(MAX_ITERS="3", ENABLE_WANDB="True", STR_LEN="3")
def test_train_not_successful(configurable_population, mocker):
    # Test solution not being found within max iterations
    population = configurable_population(elitism=0, stopping_criteria_fn=_stopping_criteria_fn)
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


@mockenv(MAX_ITERS="3", ENABLE_WANDB="True", STR_LEN="3")
def test_train_successful(configurable_population, mocker):
    # Test solution being found after first iteration
    population = configurable_population(elitism=0, stopping_criteria_fn=_stopping_criteria_fn)
    id = 1
    mocked_generate_initial_population = mocker.patch('src.generic.model.Population.generate_initial_population')
    mocked_perform_selection = mocker.patch('src.generic.model.Population.perform_selection')
    mocked_perform_crossover = mocker.patch('src.generic.model.Population.perform_crossover')
    mocked_perform_mutation = mocker.patch('src.generic.model.Population.perform_mutation')
    mocked_get_winner = mocker.patch('src.generic.model.Population.get_winner')
    mocked_get_winner.return_value = (Solution(chromosome=np.asarray([1, 1, 1]), fitness=3), 3)
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

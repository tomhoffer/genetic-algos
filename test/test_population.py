import copy

from src.model import Solution


def test_get_winner(population_with_growing_fitness, mocker):
    mocked_refresh_fitness = mocker.patch('src.model.Population.refresh_fitness')
    winner, fitness = population_with_growing_fitness.get_winner()

    mocked_refresh_fitness.assert_called_once()
    assert winner.fitness == fitness
    assert fitness == max(member.fitness for member in population_with_growing_fitness.members)


def test_perform_selection(population_with_growing_fitness, mocker):
    mocked_refresh_fitness = mocker.patch('src.model.Population.refresh_fitness')
    old_members = copy.deepcopy(population_with_growing_fitness.members)
    population_with_growing_fitness.perform_selection()
    mocked_refresh_fitness.assert_called_once()
    assert not population_with_growing_fitness.members == old_members


def test_perform_mutation(population_with_growing_fitness, mocker):
    mocked_refresh_fitness = mocker.patch('src.model.Population.refresh_fitness')
    old_members = copy.deepcopy(population_with_growing_fitness.members)
    population_with_growing_fitness.perform_mutation()
    mocked_refresh_fitness.assert_called_once()
    assert not population_with_growing_fitness.members == old_members


def test_perform_crossover(population_with_growing_fitness, mocker):
    mocked_refresh_fitness = mocker.patch('src.model.Population.refresh_fitness')
    old_members = copy.deepcopy(population_with_growing_fitness.members)
    population_with_growing_fitness.perform_crossover()
    mocked_refresh_fitness.assert_called_once()
    assert not population_with_growing_fitness.members == old_members


def test_generate_initial_population(population_with_growing_fitness, mocker):
    mocked_refresh_fitness = mocker.patch('src.model.Population.refresh_fitness')
    old_members = copy.deepcopy(population_with_growing_fitness.members)

    # Re-generate population again
    population_with_growing_fitness.generate_initial_population()

    mocked_refresh_fitness.assert_called_once()
    assert not population_with_growing_fitness.members == old_members


def test_train_not_successful(population_with_growing_fitness, mocker):
    # Test solution not being found within max iterations

    max_iters = 3
    id = 1
    mocked_generate_initial_population = mocker.patch('src.model.Population.generate_initial_population')
    mocked_perform_selection = mocker.patch('src.model.Population.perform_selection')
    mocked_perform_crossover = mocker.patch('src.model.Population.perform_crossover')
    mocked_perform_mutation = mocker.patch('src.model.Population.perform_mutation')
    mocked_get_winner = mocker.patch('src.model.Population.get_winner')
    mocked_get_winner.return_value = (Solution(chromosome="101", fitness=0), 0)
    mocked_wandb_init = mocker.patch('wandb.init')
    mocked_wandb_log = mocker.patch('wandb.log')
    mocked_os_environ_get = mocker.patch('os.environ.get')
    mocked_os_environ_get.return_value = max_iters

    winner, success, process_id = population_with_growing_fitness.train(id=id)

    assert mocked_generate_initial_population.call_count == 1
    assert mocked_wandb_init.call_count == 1
    assert mocked_wandb_log.call_count == max_iters
    assert mocked_perform_selection.call_count == max_iters
    assert mocked_perform_crossover.call_count == max_iters
    assert mocked_perform_mutation.call_count == max_iters
    assert mocked_get_winner.call_count == max_iters
    assert mocked_get_winner.call_count == max_iters

    assert winner == Solution(chromosome="101", fitness=0)
    assert success is False
    assert process_id == id


def test_train_successful(population_with_growing_fitness, mocker):
    # Test solution being found after first iteration

    max_iters = 3
    id = 1
    mocked_generate_initial_population = mocker.patch('src.model.Population.generate_initial_population')
    mocked_perform_selection = mocker.patch('src.model.Population.perform_selection')
    mocked_perform_crossover = mocker.patch('src.model.Population.perform_crossover')
    mocked_perform_mutation = mocker.patch('src.model.Population.perform_mutation')
    mocked_get_winner = mocker.patch('src.model.Population.get_winner')
    mocked_get_winner.return_value = (Solution(chromosome="111", fitness=3), 3)
    mocked_wandb_init = mocker.patch('wandb.init')
    mocked_wandb_log = mocker.patch('wandb.log')
    mocked_os_environ_get = mocker.patch('os.environ.get')
    mocked_os_environ_get.return_value = max_iters

    winner, success, process_id = population_with_growing_fitness.train(id=id)

    assert mocked_generate_initial_population.call_count == 1
    assert mocked_wandb_init.call_count == 1
    assert mocked_wandb_log.call_count == 1
    assert mocked_perform_selection.call_count == 1
    assert mocked_perform_crossover.call_count == 1
    assert mocked_perform_mutation.call_count == 1
    assert mocked_get_winner.call_count == 1
    assert mocked_get_winner.call_count == 1

    assert winner == Solution(chromosome="111", fitness=3)
    assert success is True
    assert process_id == id

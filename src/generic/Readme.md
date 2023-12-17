# Generic library for Genetic algorithms

This directory contains a generic implementation of all steps used in genetic algorithms, including:

- Selection methods (`src/generic/selction.py`)
- Crossover methods (`src/generic/crossover.py`)
- Mutation methods (`src/generic/mutation.py`)
- Training logic (`src/generic/model.py`)
- Parallel executor with Redis caching to speed-up training with complex fitness functions (`src/generic/executor.py`)

Unit tests are stored in `/test/generic`.

# Install dependencies

`pip insall -r requirements.txt`

# Usage

We will explain the usage on solving
the [one-max problem](https://subscription.packtpub.com/book/data/9781838557744/5/ch05lvl1sec28/the-onemax-problem). The
OneMax task is to find the binary string of a given length that maximizes the sum of its digits.

## 1. Define your population generator function

```
# Generates a random population of lists containing zeros and ones
def initial_population_generator() -> List[Solution]:
    result: List[Solution] = []
    for _ in range(int(os.environ.get("POPULATION_SIZE"))):
        el = np.random.choice([0, 1], size=5)
        result.append(Solution(chromosome=el))
    return result
```

## 2. Define your fitness function

In this case fitness function will be simple. The more ones the list contains, the higher the fitness score.

```
def fitness(chromosome: np.ndarray) -> int:
    return chromosome.sum()
```

## 3. Define your stopping criteria function

```
def stopping_criteria_fn(solution: Solution) -> bool:
    return solution.fitness == 5
```

## 4. Define your chromosome validator function

We will keep it dummy for explanation purposes.

```
def chromosome_validator_fn(chromosome: np.ndarray):
    return True
```

## 5. Specify parameters of your algorithm using the Hyperparams class

- fitness_fn: Function to compute the fitness logic
- initial_population_generator_fn: Function to generate initial population of solutions
- mutation_fn: Function containing the chromosome mutation logic
- selection_fn: Function performing the selection
- crossover_fn: Function performing the chromosome crossover operation
- stopping_criteria_fn: Function returning `True` once stopping criteria are met.
- chromosome_validator_fn: Function returning `True` if chromosome is valid.
- population_size: Size of the population.
- elitism: Number of elite individuals surviving into the next generation.

See `src.generic.model.Hyperparams` for particular type definitions.

```
params = Hyperparams(crossover_fn=Crossover.two_point,
                         initial_population_generator_fn=initial_population_generator,
                         mutation_fn=Mutation.flip_bit,
                         selection_fn=Selection.tournament,
                         fitness_fn=fitness, population_size=int(os.environ.get("POPULATION_SIZE")), elitism=5,
                         stopping_criteria_fn=stopping_criteria_fn, chromosome_validator_fn=chromosome_validator_fn)
```

## 6. Run the parallel search for solution

```
TrainingExecutor.run_parallel(params)
```

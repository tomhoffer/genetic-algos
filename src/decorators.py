from src.model import Solution, Population


def validate_chromosome_length(func):
    """
    Decorator to validate that chromosome length of parentA == chromosome length of parentB
    """

    def wrapper(parent1: Solution, parent2: Solution, *args, **kwargs):
        if len(parent1.chromosome) != len(parent2.chromosome):
            raise ValueError(f"Gene length does not match! Parents: {parent1}, {parent2}")

        return func(parent1, parent2, *args, **kwargs)

    return wrapper


def validate_population_length(func):
    """
        Decorator to validate that population is not empty
        """

    def wrapper(population: Population, *args, **kwargs):
        if len(population.members) < 1:
            raise ValueError(f"Population size lower than 1! Actual: {len(population.members)}")

        return func(population, *args, **kwargs)

    return wrapper

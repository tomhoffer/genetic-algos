from src.generic.model import Solution, Population


def validate_chromosome_length(func):
    """
    Decorator to validate that chromosome length of parentA == chromosome length of parentB
    """

    def wrapper(parent1: Solution, parent2: Solution, *args, **kwargs):
        if parent1.chromosome.size != parent2.chromosome.size:
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


def validate_parents_chromosome_type(type: any):
    """
    Decorator to validate that both parent's chromosomes are of desired type
    """

    def wrap(f):
        def wrapper(parent1: Solution, parent2: Solution, *args, **kwargs):
            if not (isinstance(parent1.chromosome, type) and isinstance(parent2.chromosome, type)):
                raise TypeError(f"Parent chromosomes do not match required type: {type}. Parents: {parent1}, {parent2}")
            return f(parent1, parent2, *args, **kwargs)

        return wrapper

    return wrap


def validate_chromosome_type(type: any):
    """
    Decorator to validate that chromosome is of desired type
    """

    def wrap(f):
        def wrapper(chromosome: any, *args, **kwargs):
            if not isinstance(chromosome, type):
                raise TypeError(f"Chromosome does not match required type: {type}. Chromosome: {chromosome}")
            return f(chromosome, *args, **kwargs)

        return wrapper

    return wrap

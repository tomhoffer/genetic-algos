import logging
import random


def mutate(sequence: str, probability=0.01) -> str:
    result = ""
    for gene in sequence:
        if random.random() < probability:
            # flip the bit
            result += str(1 - int(gene))
            logging.debug(f"Mutation probability hit! Mutating gene: {sequence}...")
        else:
            result += gene
    return result

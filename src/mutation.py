import logging
import random


class Mutation:

    @staticmethod
    def flip_bit(sequence: str, probability=0.01) -> str:
        result = ""
        for gene in sequence:
            if random.random() < probability:
                # flip the bit
                result += str(1 - int(gene))
                logging.debug(f"Mutation probability hit! Mutating gene: {sequence}...")
            else:
                result += gene
        return result

    @staticmethod
    def swap(sequence: str, probability=0.01) -> str:
        def make_swap(s: str, index1: int, index2: int) -> str:
            new_s = list(s)
            new_s[index1], new_s[index2] = new_s[index2], new_s[index1]
            return "".join(new_s)

        result = sequence

        if random.random() < probability:
            swap_pos_1 = random.randint(0, len(sequence) - 1)
            swap_pos_2 = random.randint(0, len(sequence) - 1)
            logging.debug(
                f"Mutation probability hit! Mutating gene via swap mutation over positions {swap_pos_1}, {swap_pos_2}: {sequence}...")

            result = make_swap(sequence, swap_pos_1, swap_pos_2)

        return result

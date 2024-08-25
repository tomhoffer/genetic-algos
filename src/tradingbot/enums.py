from enum import IntEnum


class Decision(IntEnum):
    STRONG_BUY = 2
    BUY = 1
    SELL = -1
    STRONG_SELL = -1
    INCONCLUSIVE = 0

from enum import IntEnum


class Decision(IntEnum):
    STRONG_BUY = 2
    BUY = 1
    SELL = -1
    STRONG_SELL = -1
    INCONCLUSIVE = 0


class SellTrigger(IntEnum):
    """
        Enum representing the type of sell trigger.
    """
    SHORT = 1
    STOP_LOSS = 2
    TAKE_PROFIT = 3

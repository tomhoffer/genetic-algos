from enum import Enum


class Decision(Enum):
    BUY = 1
    SELL = -1
    INCONCLUSIVE = 0


class Indicators:

    @staticmethod
    def decide_mfi(mfi_value: float) -> Decision:
        """
        Money flow index: https://www.investopedia.com/terms/m/mfi.asp
        return: Signal to buy or sell based on MFI
        """
        if mfi_value >= 80:
            return Decision.SELL
        if mfi_value <= 20:
            return Decision.BUY
        return Decision.INCONCLUSIVE

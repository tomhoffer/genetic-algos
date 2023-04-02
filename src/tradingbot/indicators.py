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

    @staticmethod
    def decide_adi(adi_start: float, adi_end: float, ticker_start: float, ticker_end: float) -> Decision:
        """
        Accumulation/Distribution Indicator: https://www.investopedia.com/terms/a/accumulationdistribution.asp
        return: Signal to buy or sell based on ADI
        :param adi_start: Value of ADI indicator at the beginning of the window (e.g. window =7d, adi_start = adi 7 days ago)
        :param adi_end: Value of ADI indicator at the end of the window (adi_start = adi now)
        :param ticker_start: Ticker price at the beginning of the window (e.g. window =7d, ticker_start = price 7 days ago)
        :param ticker_end: Ticker price at the end of the window (ticker_end = current price)
        """

        trend_adi = adi_end - adi_start
        trend_price = ticker_end - ticker_start

        if trend_price <= 0 < trend_adi:
            return Decision.BUY
        if trend_price >= 0 > trend_adi:
            return Decision.SELL
        return Decision.INCONCLUSIVE

    @staticmethod
    def decide_cmf(cmf_value: float) -> Decision:
        """
        Chaikin Money Flow indicator: https://www.investopedia.com/ask/answers/071414/whats-difference-between-chaikin-money-flow-cmf-and-money-flow-index-mfi.asp
        :return: Signal to buy or sell based on CMF
        """

        if cmf_value > 0.2:
            return Decision.SELL
        if cmf_value < -0.2:
            return Decision.BUY
        return Decision.INCONCLUSIVE

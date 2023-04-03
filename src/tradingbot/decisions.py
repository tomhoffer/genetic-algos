from enum import Enum


class Decision(Enum):
    BUY = 1
    SELL = -1
    INCONCLUSIVE = 0


class Indicators:

    # TODO Current implementation is based on binary decisions of each indicators.
    #  Indicators however give us information on how strong the buy/sell signal is.
    #  Return the signal strength instead of binary decision.

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

    @staticmethod
    def decide_em(em_value: float) -> Decision:  # TODO make thresholds trainable parameters
        """
        Ease of movement indicator: https://www.investopedia.com/terms/e/easeofmovement.asp
        :return: Signal to buy or sell based on EM
        """
        if em_value > 0.1:
            return Decision.BUY
        if em_value < -0.1:
            return Decision.SELL
        return Decision.INCONCLUSIVE

    @staticmethod
    def decide_vpt(vpt_value: float, vpt_sma_value: float) -> Decision:
        """
        Volume Price Trend Indicator: https://www.investopedia.com/terms/v/vptindicator.asp
        :param vpt_value:
        :param vpt_sma_value: Simple moving average of the VPT
        :return: Signal to buy or sell based on VPT and its SMA
        """
        if vpt_value == vpt_sma_value:
            return Decision.INCONCLUSIVE
        return Decision.BUY if vpt_value > vpt_sma_value else Decision.SELL

    @staticmethod
    def decide_macd(macd_value: float, macd_signal_value: float) -> Decision:
        """
        Moving Average Convergence Divergence indicator: https://www.investopedia.com/ask/answers/122414/what-moving-average-convergence-divergence-macd-formula-and-how-it-calculated.asp
        :return: Signal to buy or sell based on MACD
        """
        if macd_value == macd_signal_value:
            return Decision.INCONCLUSIVE
        return Decision.BUY if macd_value > macd_signal_value else Decision.SELL

    @staticmethod
    def decide_nvi(nvi_value: float, nvi_ema_255_value: float) -> Decision:
        """
        Negative volume indicator: https://www.warriortrading.com/negative-volume-index-nvi-indicator/
        :param nvi_value:
        :param nvi_ema_255_value: 255d Exponential moving average of NVI
        :return: Signal to buy or sell based on NVI
        """
        if nvi_value == nvi_ema_255_value:
            return Decision.INCONCLUSIVE
        return Decision.BUY if nvi_value > nvi_ema_255_value else Decision.SELL

    @staticmethod
    def decide_vwap(vwap_value: float, ticker_price: float) -> Decision:
        """
        Volume-Weighted Average Price https://www.investopedia.com/terms/v/vwap.asp
        :param vwap_value:
        :param ticker_price:
        :return: Signal to buy or sell based on VWAP and current stock price
        """
        if ticker_price == vwap_value:
            return Decision.INCONCLUSIVE
        return Decision.BUY if ticker_price < vwap_value else Decision.SELL


class Sentiment:
    @staticmethod
    def decide_sentiment(sentiment_value: float) -> Decision:
        if sentiment_value == 0.5:
            return Decision.INCONCLUSIVE
        return Decision.BUY if sentiment_value > 0.5 else Decision.SELL

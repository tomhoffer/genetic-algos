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

    @staticmethod
    def decide_death_cross(sma_50_value: float, sma_200_value: float) -> Decision:
        """
        Death cross pattern: https://www.investopedia.com/terms/d/deathcross.asp
        :param sma_50_value: Simple moving average (50d)
        :param sma_200_value: Simple moving average (200d)
        :return: Decision to sell based on death cross pattern
        """
        return Decision.SELL if sma_50_value <= sma_200_value else Decision.INCONCLUSIVE

    @staticmethod
    def decide_golden_cross(sma_50_value: float, sma_200_value: float) -> Decision:
        """
        Golden cross pattern: https://www.investopedia.com/terms/g/goldencross.asp
        :param sma_50_value: Simple moving average (50d)
        :param sma_200_value: Simple moving average (200d)
        :return: Decision to buy based on golden cross pattern
        """
        return Decision.BUY if sma_50_value >= sma_200_value else Decision.INCONCLUSIVE

    @staticmethod
    def decide_sma_fast(sma_12_value: float, ticker_price: float) -> Decision:
        """
        Simple moving average (12d): https://www.indiainfoline.com/knowledge-center/trading-account/what-is-a-simple-moving-average-trading-strategy
        :param sma_12_value: Simple moving average (12d)
        :param ticker_price:
        :return: Decision to buy or sell based on Simple moving average (12d)
        """
        if sma_12_value == ticker_price:
            return Decision.INCONCLUSIVE
        return Decision.BUY if ticker_price > sma_12_value else Decision.SELL

    @staticmethod
    def decide_sma_slow(sma_26_value: float, ticker_price: float) -> Decision:
        """
        Simple moving average (26d): https://www.indiainfoline.com/knowledge-center/trading-account/what-is-a-simple-moving-average-trading-strategy
        :param sma_26_value: Simple moving average (26d)
        :param ticker_price:
        :return: Decision to buy or sell based on Simple moving average (26d)
        """
        if sma_26_value == ticker_price:
            return Decision.INCONCLUSIVE
        return Decision.BUY if ticker_price > sma_26_value else Decision.SELL

    @staticmethod
    def decide_ema_20_vs_50(ema_20_value: float, ema_50_value: float, ticker_price: float) -> Decision:
        """
        Exponential moving average 20d vs 50d: https://tradingstrategyguides.com/exponential-moving-average-strategy/
        :param ema_20_value:
        :param ema_50_value:
        :return: Decision to buy or sell based on EMA 20d, EMA 50d and the current ticker price
        """
        if ticker_price > ema_20_value > ema_50_value:
            return Decision.BUY

        if ema_50_value > ema_20_value > ticker_price:
            return Decision.SELL

        return Decision.INCONCLUSIVE

    @staticmethod
    def decide_adx(adx_value: float, adx_pos_value: float, adx_neg_value: float) -> Decision:
        """
        Average Directional Index https://www.investopedia.com/terms/a/adx.asp
        :param adx_value:
        :param adx_pos_value:
        :param adx_neg_value:
        :return: Decision to buy or sell based on ADX indicator
        """

        is_trend_strong = adx_value > 25
        is_trend_weak = adx_value < 20
        is_trend_inconclusive = not (is_trend_strong or is_trend_weak)

        if is_trend_weak:
            return Decision.INCONCLUSIVE

        if (is_trend_strong or is_trend_inconclusive) and adx_pos_value > adx_neg_value:
            return Decision.BUY

        if (is_trend_strong or is_trend_inconclusive) and adx_neg_value > adx_pos_value:
            return Decision.SELL

    @staticmethod
    def decide_vi(vortex_diff_value: float) -> Decision:
        """
        Vortex indicator https://www.investopedia.com/terms/v/vortex-indicator-vi.asp
        :param vortex_diff_value:
        :return: Decision to buy or sell based on Vortex indicator
        """
        if vortex_diff_value == 0:
            return Decision.INCONCLUSIVE
        return Decision.BUY if vortex_diff_value > 0 else Decision.SELL

    @staticmethod
    def decide_trix(trix_value: float) -> Decision:
        """
        Triple Exponential Average https://www.investopedia.com/terms/t/trix.asp
        :param trix_value:
        :return: Decision based on TRIX indicator
        """
        if trix_value == 0:
            return Decision.INCONCLUSIVE
        return Decision.BUY if trix_value > 0 else Decision.SELL

class Sentiment:
    @staticmethod
    def decide_sentiment(sentiment_value: float) -> Decision:
        if sentiment_value == 0.5:
            return Decision.INCONCLUSIVE
        return Decision.BUY if sentiment_value > 0.5 else Decision.SELL

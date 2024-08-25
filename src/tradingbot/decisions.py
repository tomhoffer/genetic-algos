import numpy as np
from typing import Dict

from src.tradingbot.config import Config
from src.tradingbot.enums import Decision


class TradingStrategies:

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
        if em_value > 0.2:
            return Decision.STRONG_BUY
        if em_value > 0.1:
            return Decision.BUY
        if em_value < -0.2:
            return Decision.STRONG_SELL
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

    @classmethod
    def decide_vpt_adx(cls, vpt_value: float, vpt_sma_value: float, adx_value: float) -> Decision:
        """
            ADX indicator confirmed by VPT indicator
                :param vpt_value:
                :param vpt_sma_value: Simple moving average of the VPT
                :return: Signal to buy or sell based on VPT and ADX
                """
        result_vpt: Decision = cls.decide_vpt(vpt_value=vpt_value, vpt_sma_value=vpt_sma_value)

        if adx_value > 25:
            return result_vpt
        return Decision.INCONCLUSIVE

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

        if is_trend_inconclusive and adx_pos_value > adx_neg_value:
            return Decision.BUY

        if is_trend_strong and adx_pos_value > adx_neg_value:
            return Decision.STRONG_BUY

        if is_trend_inconclusive and adx_neg_value > adx_pos_value:
            return Decision.SELL

        if is_trend_strong and adx_neg_value > adx_pos_value:
            return Decision.STRONG_SELL

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

    @staticmethod
    def decide_sentiment(sentiment_value: float) -> Decision:
        if np.isnan(sentiment_value):
            return Decision.INCONCLUSIVE

        if sentiment_value == 0.5:
            return Decision.INCONCLUSIVE
        if sentiment_value > 0.7:
            return Decision.STRONG_BUY
        if sentiment_value > 0.5:
            return Decision.BUY
        if sentiment_value < 0.3:
            return Decision.STRONG_SELL
        if sentiment_value < 0.5:
            return Decision.SELL

    @staticmethod
    def decide_dpo(dpo_value: float) -> Decision:
        """
        Detrended Price Oscillator https://www.investopedia.com/terms/d/detrended-price-oscillator-dpo.asp
        :param dpo_value: DPO value
        :return: Decision based on DPO
        """
        if -1 < dpo_value < 1:
            return Decision.INCONCLUSIVE
        return Decision.BUY if dpo_value > 0 else Decision.SELL

    @staticmethod
    def decide_rsi(rsi_value: float) -> Decision:
        """
        Relative strength index https://www.investopedia.com/terms/r/rsi.asp
        :param rsi_value: RSI value
        :return: Decision to buy or sell based on RSI
        """
        if rsi_value >= 70:
            return Decision.SELL
        elif rsi_value <= 30:
            return Decision.BUY
        return Decision.INCONCLUSIVE

    @staticmethod
    def decide_stoch_rsi(stoch_rsi_value: float) -> Decision:
        """
        Stochastic Relative strength index https://www.investopedia.com/terms/s/stochrsi.asp
        :param stoch_rsi_value: Stochastic RSI value
        :return: Decision to buy or sell based on Stochastic RSI
        """
        if stoch_rsi_value >= 80:
            return Decision.SELL
        elif stoch_rsi_value <= 20:
            return Decision.BUY
        return Decision.INCONCLUSIVE

    @staticmethod
    def decide_tsi(tsi_value: float) -> Decision:
        """
        The True Strength Index https://www.investopedia.com/terms/t/tsi.asp
        :param tsi_value: TSI value
        :return: Decision to buy or sell based on TSI value
        """
        if tsi_value > 0:
            return Decision.BUY
        elif tsi_value < 0:
            return Decision.SELL
        return Decision.INCONCLUSIVE

    @staticmethod
    def decide_tsi_signal(tsi_value: float, tsi_ema_12_value: float) -> Decision:
        """
        The True Strength Index using EMA-12 as its signal line https://www.investopedia.com/terms/t/tsi.asp
        :param tsi_value: TSI value
        :param tsi_ema_12_value: EMA-12 of TSI
        :return: Decision to buy or sell based on TSI and its EMA-12 value
        """
        if tsi_value == tsi_ema_12_value:
            return Decision.INCONCLUSIVE
        return Decision.BUY if tsi_value > tsi_ema_12_value else Decision.SELL

    @staticmethod
    def decide_uo(uo_value: float) -> Decision:
        """
        Ultimate Oscillator https://www.investopedia.com/terms/u/ultimateoscillator.asp
        :param uo_value: UO value
        :return: Decision to buy or sell based on UO
        """
        if uo_value >= 70:
            return Decision.SELL
        elif uo_value <= 30:
            return Decision.BUY
        return Decision.INCONCLUSIVE

    @staticmethod
    def decide_so(so_value: float) -> Decision:
        """
        Stochastic Oscillator https://www.investopedia.com/terms/s/stochasticoscillator.asp
        :param so_value: Stochastic Oscillator value
        :return: Decision to buy or sell based on SO
        """
        if so_value >= 80:
            return Decision.SELL
        elif so_value <= 20:
            return Decision.BUY
        return Decision.INCONCLUSIVE

    @staticmethod
    def decide_williams(williams_value: float) -> Decision:
        """
        Williams %R https://www.investopedia.com/terms/w/williamsr.asp
        :param williams_value: Williams %R value
        :return: Decision to buy or sell based on Williams %R
        """
        if williams_value >= -20:
            return Decision.SELL
        elif williams_value <= -80:
            return Decision.BUY
        return Decision.INCONCLUSIVE


    def perform_decisions_for_row(self, row: np.array, row_np_index: Dict) -> Dict:
        result_obj = {}
        ticker_name = Config.get_value("TRADED_TICKER_NAME")
        ticker_price: float = row[row_np_index[f"{ticker_name}_Adj Close"]]

        result_obj['mfi'] = self.decide_mfi(row[row_np_index['volume_mfi']])
        result_obj['adi'] = self.decide_adi(adi_start=row[row_np_index['volume_adi_7d_ago']],
                                            adi_end=row[row_np_index['volume_adi']],
                                            ticker_start=row[row_np_index[f"{ticker_name}_Close_7d_ago"]],
                                            ticker_end=ticker_price)
        result_obj['cmf'] = self.decide_cmf(row[row_np_index['volume_cmf']])
        result_obj['em'] = self.decide_cmf(row[row_np_index['volume_em']])
        result_obj['vpt'] = self.decide_vpt(vpt_value=row[row_np_index['volume_vpt']],
                                            vpt_sma_value=row[row_np_index['volume_vpt_sma']])
        result_obj['vpt_adx'] = self.decide_vpt_adx(vpt_value=row[row_np_index['volume_vpt']],
                                                    vpt_sma_value=row[row_np_index['volume_vpt_sma']],
                                                    adx_value=row[row_np_index['trend_adx']])
        result_obj['macd'] = self.decide_macd(macd_value=row[row_np_index['trend_macd']],
                                              macd_signal_value=row[row_np_index['trend_macd_signal']])
        result_obj['nvi'] = self.decide_nvi(nvi_value=row[row_np_index['volume_nvi']],
                                            nvi_ema_255_value=row[row_np_index['trend_nvi_ema_255']])
        result_obj['vwap'] = self.decide_vwap(vwap_value=row[row_np_index['volume_vwap']], ticker_price=ticker_price)
        result_obj['death_cross'] = self.decide_death_cross(sma_50_value=row[row_np_index['trend_sma_50']],
                                                            sma_200_value=row[row_np_index['trend_sma_200']])
        result_obj['golden_cross'] = self.decide_golden_cross(sma_50_value=row[row_np_index['trend_sma_50']],
                                                              sma_200_value=row[row_np_index['trend_sma_200']])
        result_obj['sma_fast'] = self.decide_sma_fast(sma_12_value=row[row_np_index['trend_sma_fast']],
                                                      ticker_price=ticker_price)
        result_obj['sma_slow'] = self.decide_sma_slow(sma_26_value=row[row_np_index['trend_sma_slow']],
                                                      ticker_price=ticker_price)
        result_obj['ema_20_vs_50'] = self.decide_ema_20_vs_50(ticker_price=ticker_price,
                                                              ema_20_value=row[row_np_index['trend_ema_20']],
                                                              ema_50_value=row[row_np_index['trend_ema_50']])
        result_obj['adx'] = self.decide_adx(adx_value=row[row_np_index['trend_adx']],
                                            adx_pos_value=row[row_np_index['trend_adx_pos']],
                                            adx_neg_value=row[row_np_index['trend_adx_neg']])
        result_obj['vi'] = self.decide_vi(vortex_diff_value=row[row_np_index['trend_vortex_ind_diff']])
        result_obj['trix'] = self.decide_trix(trix_value=row[row_np_index['trend_trix']])
        result_obj['dpo'] = self.decide_dpo(dpo_value=row[row_np_index['trend_dpo']])
        result_obj['rsi'] = self.decide_rsi(rsi_value=row[row_np_index['momentum_rsi']])
        result_obj['stoch_rsi'] = self.decide_stoch_rsi(stoch_rsi_value=row[row_np_index['momentum_stoch_rsi']])
        result_obj['tsi'] = self.decide_tsi(tsi_value=row[row_np_index['momentum_tsi']])
        result_obj['uo'] = self.decide_uo(uo_value=row[row_np_index['momentum_uo']])
        result_obj['so'] = self.decide_so(so_value=row[row_np_index['momentum_stoch']])
        result_obj['williams'] = self.decide_williams(williams_value=row[row_np_index['momentum_wr']])
        result_obj['tsi_signal'] = self.decide_tsi_signal(tsi_value=row[row_np_index['momentum_tsi']],
                                                          tsi_ema_12_value=row[row_np_index['momentum_tsi_ema_12']])

        try:
            result_obj['sentiment'] = self.decide_sentiment(sentiment_value=row[row_np_index['sentiment']])
        except (KeyError, IndexError):  # Missing sentiment column
            result_obj['sentiment'] = Decision.INCONCLUSIVE
        return result_obj

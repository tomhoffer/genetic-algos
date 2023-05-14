# Tradingbot

Tradingbot is an algorithm able to trade a certain asset in real-time.
The algorithm learns the importance (weights) of individual trading strategies to use in order to make a decision to buy
or sell.

# Prerequisites

Tradingbot requires offline stock trading data to be downloaded and stored in `/data`.

# Chromosome structure

```
[weight_of_trading_strategy: float <0,1>]
```

Configurable parameters (.env):

- Total amount of money to trade: BUDGET
- Starting timestamp: START_TIMESTAMP
- Date of evaluation: END_TIMESTAMP
- Backtest period start timestamp: BACKTEST_START_TIMESTAMP
- Backtest period end timestamp BACKTEST_END_TIMESTAMP
- TRADED_TICKER_NAME: Name of the traded ticker (used to read training data at `data/data-$TRADED_TICKER_NAME.csv`)
- RETURN_GLOBAL_WINNER: If True, best individual from all epochs is returned. If False, best individual from the last
  epoch is returned

See `config.py` for data types of parameters mentioned above.

# Run via Docker-compose
```
docker-compose up
```
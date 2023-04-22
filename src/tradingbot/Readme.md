# Investobot

Tradingbot is an algorithm able to trade a certain asset in real-time.
The algorithm learns the importance (weights) of individual trading strategies to use in order to make a decision to buy or sell.

# Prerequisites
Tradingbot requires offline stock trading data to be downloaded and stored in ./data.csv.


# Chromosome structure

```
[weight_of_trading_strategy: float <0,1>]
```

Configurable parameters (.env):
- Total amount of money to trade: BUDGET
- Size of each trade: TRADE_SIZE
- Starting timestamp: START_TIMESTAMP
- Date of evaluation: END_TIMESTAMP

See `config.py` for data types of parameters mentioned above.
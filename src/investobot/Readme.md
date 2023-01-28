# Investobot

Investobot is an algorithm able to invest in various assets.
The algorithm learns:

- Which assets to invest
- What proportion of money to invest into each asset
- When to invest into particular assets

# Prerequisites
Investobot requires offline stock trading data to be downloaded and stored in ./data.csv.


# Chromosome structure

```
[
    [
        ticker: int, // (asset id)
        amount: int, // amount of cash invested
        timestamp: long // timestamp of investment
    ]...
]
```

Configurable parameters (.env):
- Number of transactions: NUM_TRANSACTIONS
- Max yearly trades: MAX_YEARLY_TRADES
- Amount of money to invest: BUDGET
- Earliest possible date for any of the investments: START_TIMESTAMP
- Date of evaluation (same for all investments): END_TIMESTAMP

See `config.py` for data types of parameters mentioned above.
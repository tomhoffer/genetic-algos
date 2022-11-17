# Investobot

Investobot is an algorithm able to invest in various assets.
The algorithm learns:

- Which assets to invest
- What proportion of money to invest into each asset
- When to invest into particular assets

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

- Chromosome max length: Hyperparameter (CHROMOSOME_MAX_LENGTH)
- Max yearly trades: Hyperparameter (MAX_YEARLY_TRADES)
- Budget: Configurable in .env
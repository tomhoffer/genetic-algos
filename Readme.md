This project demonstrates and utilizes genetic algorithms to trade stocks and commodities on the stock market.

## Project structure
- `src/generic`: Contains the generic implementation of all genetic algorithm steps, applicable on any problem solvable by genetic algorithms, including the framework for training and parallelization. More in `src/generic/Readme.md`
- `src/investobot`: Contains implementation of a bot which learns how to distribute weights of particular investments in its portfolio. More in `src/investobot/Readme.md`
- `src/tradingbot`: Contains implementation of a tradingbot which learns the importance (weights) of individual trading strategies to use when making a decision to buy
or sell a certain asset on stock market. More in `src/tradingbot/Readme.md`

# Install dependencies

```
pip install -r requirements.txt
```

# Initialize wandb

```
wandb login
```

# Run the code
Here are the basic make commands. For more supported make commands, refer to `Makefile`. Installing Docker and docker-compose is a prerequisite.


`make run-test` to run all tests in a docker container

`make run-test-local` to run all tests locally

`make run-tradingbot` to run the tradingbot in a docker container

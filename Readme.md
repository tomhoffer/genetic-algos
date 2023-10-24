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

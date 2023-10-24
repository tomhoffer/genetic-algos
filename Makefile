build-tradingbot:
	docker build src/tradingbot

run-tradingbot: build-tradingbot
	docker-compose up tradingbot

run-test:
	docker-compose up test

run-test-local:
	python -m pytest


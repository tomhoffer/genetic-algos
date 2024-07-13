build-tradingbot:
	docker build src/tradingbot

run-tradingbot: build-tradingbot
	docker-compose up tradingbot

run-tradingbot-test:
	docker-compose up test-tradingbot

run-generic-test:
	docker-compose up test-generic

run-test-local:
	python -m pytest


build-tradingbot:
	docker build src/tradingbot

run-tradingbot: build-tradingbot
	docker-compose up tradingbot

run-test:
	python -m pytest
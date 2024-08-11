build-tradingbot:
	docker-compose build tradingbot

run-tradingbot: build-tradingbot
	docker-compose up tradingbot

run-tradingbot-test:
	docker-compose up test-tradingbot

build-tradingbot-test:
	docker-compose build test-tradingbot

run-generic-test:
	docker-compose up test-generic

run-test-local:
	python -m pytest

build-tradingbot-server:
	docker-compose build tradingbot-server

run-tradingbot-server: build-tradingbot-server
	docker-compose up tradingbot-server
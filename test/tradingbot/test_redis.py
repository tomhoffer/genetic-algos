from src.tradingbot.decorators import redis_cache
from src.tradingbot.redis_connector import connect


@redis_cache
def dummy_fn(arg):
    return 1


def test_redis_integration():
    # Test successful connection to Redis server
    connection = connect()
    assert connection.ping()

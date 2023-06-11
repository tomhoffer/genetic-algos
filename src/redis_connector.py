import redis


def connect():
    return redis.Redis(host='localhost', port=6379, decode_responses=True)

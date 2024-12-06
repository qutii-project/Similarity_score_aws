import redis

redis_host = "similarity-score-cache-9wmkxx.serverless.eun1.cache.amazonaws.com"
redis_port = 6379

try:
    r = redis.StrictRedis(host=redis_host, port=redis_port, decode_responses=True)
    r.ping()
    print("Connected to Redis!")
except redis.ConnectionError as e:
    print(f"Error connecting to Redis: {e}")
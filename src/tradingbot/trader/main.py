import http.client
import json
import time
import urllib

from src.tradingbot.enums import Decision
from src.tradingbot.trader.config import Config


def send_push_notification(trade_result: Decision):
    if trade_result == Decision.BUY:
        message = f"Advice from tradingbot: Buy AAPL!"
    elif trade_result == Decision.SELL:
        message = f"Advice from tradingbot: Sell AAPL!"
    else:
        message = "Mixed feelings today, let's try tomorrow..."

    conn = http.client.HTTPSConnection("api.pushover.net:443")
    conn.request("POST", "/1/messages.json",
                 urllib.parse.urlencode({
                     "token": Config.get_value("APP_KEY"),
                     "user": Config.get_value("USER_KEY"),
                     "message": message,
                 }), {"Content-type": "application/x-www-form-urlencoded"})
    conn.getresponse()


def get_tradingbot_decision() -> Decision:
    conn = http.client.HTTPConnection("tradingbot-server:80")
    conn.request("GET", "/decision/")
    res = conn.getresponse().read()
    res = json.loads(res.decode("utf-8"))
    return res["decision"]


if __name__ == "__main__":
    decision: Decision = get_tradingbot_decision()
    send_push_notification(decision)

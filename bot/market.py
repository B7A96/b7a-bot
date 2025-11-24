import requests

COIN_MAP = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "DOGE": "dogecoin",
    "TON": "the-open-network",
}


def get_price_usd(symbol: str) -> float | None:
    symbol = symbol.upper()
    coin_id = COIN_MAP.get(symbol)
    if not coin_id:
        return None

    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": coin_id, "vs_currencies": "usd"}

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    return data.get(coin_id, {}).get("usd")


def generate_demo_signal() -> dict:
    return {
        "symbol": "BTC",
        "direction": "LONG",
        "entry": "85,000 - 86,000",
        "take_profit": "90,000",
        "stop_loss": "83,500",
    }

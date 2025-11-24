import requests

BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price?symbol={}"

def get_price(symbol: str):
    """
    Fetches the real-time price from Binance.
    
    symbol: e.g. 'BTC', 'ETH', 'SOL'
    """

    symbol = symbol.upper() + "USDT"

    try:
        response = requests.get(BINANCE_PRICE_URL.format(symbol))
        data = response.json()

        if "price" in data:
            return float(data["price"])
        else:
            return None

    except Exception as e:
        print("Error fetching price:", e)
        return None

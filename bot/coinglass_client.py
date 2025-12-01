# bot/coinglass_client.py
import os
import requests
from typing import Dict, Any

BASE_URL = "https://open-api.coinglass.com/public/v2"  # تأكد من الـ base من الدوكيومنت الرسمي
API_KEY = os.getenv("COINGLASS_API_KEY")


class CoinGlassError(Exception):
    pass


def _get(path: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not API_KEY:
        raise CoinGlassError("COINGLASS_API_KEY is not set")

    headers = {
        "coinglassSecret": API_KEY
    }
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, headers=headers, params=params, timeout=10)

    if resp.status_code != 200:
        raise CoinGlassError(f"HTTP {resp.status_code}: {resp.text}")

    data = resp.json()
    return data

"""
Crypto domain for binary price prediction markets.

Focuses on: "Will BTC/ETH be above $X by date Y?"

External data: Historical prices, volatility from crypto data container.

Usage:
    from polymarket_agents.domains.crypto import CryptoAgent
    agent = CryptoAgent(price_source=my_container)
    recommendations = agent.run()
"""

from .agent import CryptoAgent, TradeRecommendation
from .models import Asset, CryptoPriceMarket, PriceDataSource, PriceSignal
from .scanner import CryptoScanner

__all__ = [
    "Asset",
    "CryptoAgent",
    "CryptoPriceMarket",
    "CryptoScanner",
    "PriceDataSource",
    "PriceSignal",
    "TradeRecommendation",
]

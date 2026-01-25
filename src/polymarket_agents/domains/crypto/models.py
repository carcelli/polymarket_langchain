"""
Crypto domain models.

Binary price prediction: "Will BTC be above $100k by March 2025?"
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Protocol
import re


class Asset(str, Enum):
    BTC = "BTC"
    ETH = "ETH"
    SOL = "SOL"
    XRP = "XRP"
    DOGE = "DOGE"


@dataclass
class PriceSignal:
    """Signal from external price data."""

    asset: Asset
    current_price: float
    price_24h_ago: float
    volatility_24h: float  # Standard deviation as percentage
    timestamp: datetime

    @property
    def change_24h(self) -> float:
        """24h price change as percentage."""
        return ((self.current_price - self.price_24h_ago) / self.price_24h_ago) * 100

    @property
    def trend(self) -> str:
        if self.change_24h > 2:
            return "bullish"
        elif self.change_24h < -2:
            return "bearish"
        return "neutral"


@dataclass
class CryptoPriceMarket:
    """
    Binary price prediction market.

    Example: "Will Bitcoin be above $100,000 on March 31?"
    """

    id: str
    question: str
    asset: Asset
    strike_price: float  # The price threshold
    expiry: datetime
    yes_price: float  # Market's current YES price (0-1)
    volume: float
    liquidity: float
    token_id: str
    event_id: str

    # Enriched from external data
    signal: Optional[PriceSignal] = None

    @property
    def implied_prob(self) -> float:
        """Market's implied probability of hitting strike."""
        return self.yes_price

    @property
    def time_to_expiry_hours(self) -> float:
        """Hours until market resolves."""
        delta = self.expiry - datetime.utcnow()
        return max(0, delta.total_seconds() / 3600)

    @property
    def distance_to_strike(self) -> Optional[float]:
        """
        Current price distance to strike as percentage.
        Positive = below strike, Negative = above strike.
        """
        if not self.signal:
            return None
        return (
            (self.strike_price - self.signal.current_price) / self.signal.current_price
        ) * 100

    def calculate_edge(self, our_prob: float) -> float:
        """
        Edge = our probability - market's implied probability.
        Positive edge = market underpricing YES.
        """
        return our_prob - self.implied_prob


class PriceDataSource(Protocol):
    """
    Protocol for external crypto price data.

    Implement this to connect to your price data container.
    """

    def get_current_price(self, asset: Asset) -> float:
        """Get current spot price."""
        ...

    def get_price_history(
        self, asset: Asset, hours: int = 24
    ) -> list[tuple[datetime, float]]:
        """Get historical prices."""
        ...

    def get_volatility(self, asset: Asset, hours: int = 24) -> float:
        """Get volatility (std dev as percentage)."""
        ...


def parse_strike_price(question: str) -> Optional[float]:
    """
    Extract strike price from market question.

    Examples:
        "Will Bitcoin be above $100,000 on March 31?" -> 100000.0
        "Bitcoin Up or Down — price to beat $92,994.26" -> 92994.26
    """
    patterns = [
        r"\$([0-9,]+\.?\d*)",  # $100,000 or $100,000.50
        r"above ([0-9,]+\.?\d*)k",  # above 100k
        r"([0-9,]+\.?\d*)\s*dollars",  # 100000 dollars
    ]

    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            price_str = match.group(1).replace(",", "")
            try:
                price = float(price_str)
                # Handle "100k" style
                if "k" in question.lower() and price < 1000:
                    price *= 1000
                return price
            except ValueError:
                continue

    return None


def parse_asset(question: str) -> Optional[Asset]:
    """
    Extract asset from market question.

    Examples:
        "Will Bitcoin be above..." -> Asset.BTC
        "ETH price prediction..." -> Asset.ETH
    """
    q = question.lower()

    if "bitcoin" in q or "btc" in q:
        return Asset.BTC
    elif "ethereum" in q or "eth" in q:
        return Asset.ETH
    elif "solana" in q or "sol" in q:
        return Asset.SOL
    elif "xrp" in q or "ripple" in q:
        return Asset.XRP
    elif "doge" in q or "dogecoin" in q:
        return Asset.DOGE

    return None

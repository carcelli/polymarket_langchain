"""
Base protocols for domain scanners.

All domain scanners implement EventScanner protocol.
Each domain defines its own Market subclass with domain-specific fields.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Market:
    """Base market from Polymarket."""

    id: str
    question: str
    yes_price: float  # 0.0 to 1.0
    volume: float
    liquidity: float
    end_date: Optional[datetime]
    event_id: str
    token_id: str  # For trading

    @property
    def no_price(self) -> float:
        return 1.0 - self.yes_price

    @property
    def implied_prob(self) -> float:
        """Market's implied probability (YES side)."""
        return self.yes_price


@dataclass
class ScanResult:
    """Result from scanning a domain."""

    markets: list[Market]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = "polymarket"

    def __len__(self) -> int:
        return len(self.markets)

    def top_by_volume(self, n: int = 10) -> list[Market]:
        return sorted(self.markets, key=lambda m: m.volume, reverse=True)[:n]

    def top_by_liquidity(self, n: int = 10) -> list[Market]:
        return sorted(self.markets, key=lambda m: m.liquidity, reverse=True)[:n]


class EventScanner(ABC):
    """
    Protocol for domain-specific event scanners.

    Each domain implements:
    - scan(): Fetch relevant markets from Polymarket
    - enrich(): Add external data (prices, stats, etc.)
    - filter_tradeable(): Return markets worth trading
    """

    @abstractmethod
    def scan(self) -> ScanResult:
        """Fetch all relevant markets for this domain."""
        pass

    @abstractmethod
    def enrich(self, markets: list[Market]) -> list[Market]:
        """
        Enrich markets with external data.

        Crypto: Add historical prices, volatility from price container.
        NBA: Add team stats, injuries from sports container.
        """
        pass

    @abstractmethod
    def filter_tradeable(
        self,
        markets: list[Market],
        min_volume: float = 1000,
        min_liquidity: float = 500,
    ) -> list[Market]:
        """
        Filter to markets worth trading.

        Criteria varies by domain but generally:
        - Sufficient volume/liquidity
        - Edge exists (our estimate != market price)
        - Risk/reward acceptable
        """
        pass


@dataclass
class Edge:
    """Calculated edge on a market."""

    market_id: str
    our_prob: float  # Our estimated probability
    market_prob: float  # Market's implied probability

    @property
    def edge(self) -> float:
        """Positive = we think YES is underpriced."""
        return self.our_prob - self.market_prob

    @property
    def side(self) -> str:
        """Which side to bet."""
        return "YES" if self.edge > 0 else "NO"

    @property
    def edge_magnitude(self) -> float:
        """Absolute edge size."""
        return abs(self.edge)

    def kelly_fraction(self, bankroll: float = 1.0) -> float:
        """
        Kelly criterion position sizing.

        f* = (bp - q) / b
        where b = odds, p = win prob, q = lose prob
        """
        if self.edge <= 0:
            return 0.0

        if self.side == "YES":
            b = (1 / self.market_prob) - 1  # Decimal odds - 1
            p = self.our_prob
        else:
            b = (1 / (1 - self.market_prob)) - 1
            p = 1 - self.our_prob

        q = 1 - p
        kelly = (b * p - q) / b if b > 0 else 0
        return max(0, min(kelly, 0.25))  # Cap at 25% of bankroll

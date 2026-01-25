"""
Application context for dependency injection.

Replaces hardcoded paths and singleton getters with configurable context.

Usage:
    from polymarket_agents.context import AppContext, get_context, set_context

    # Configure once at startup
    ctx = AppContext(
        db_path="data/markets.db",
        price_source=my_price_api,
        sports_source=my_sports_api,
    )
    set_context(ctx)

    # Use anywhere
    from polymarket_agents.context import get_context
    ctx = get_context()
    db = ctx.get_memory_manager()
"""

from dataclasses import dataclass, field
from typing import Optional, Any, Protocol
import os


class PriceSource(Protocol):
    """Protocol for crypto price data."""

    def get_current_price(self, asset: str) -> float: ...
    def get_price_history(self, asset: str, hours: int) -> list: ...
    def get_volatility(self, asset: str, hours: int) -> float: ...


class SportsSource(Protocol):
    """Protocol for sports data."""

    def get_team_stats(self, team: str) -> Any: ...
    def get_player_stats(self, player: str) -> Any: ...
    def get_injuries(self, team: str) -> list[str]: ...


@dataclass
class AppContext:
    """
    Application-wide context for dependency injection.

    All configurable dependencies live here instead of being hardcoded.
    """

    # Database
    db_path: str = field(
        default_factory=lambda: os.getenv("DATABASE_PATH", "data/markets.db")
    )

    # External data sources (implement Protocol or leave None for defaults)
    price_source: Optional[PriceSource] = None
    sports_source: Optional[SportsSource] = None

    # API configuration
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    clob_api_url: str = "https://clob.polymarket.com"

    # Model configuration
    default_model: str = field(
        default_factory=lambda: os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    )

    # Cached instances (lazy-initialized)
    _memory_manager: Optional[Any] = field(default=None, repr=False)
    _gamma_client: Optional[Any] = field(default=None, repr=False)
    _polymarket_client: Optional[Any] = field(default=None, repr=False)

    def get_memory_manager(self):
        """Get or create MemoryManager instance."""
        if self._memory_manager is None:
            from polymarket_agents.memory.manager import MemoryManager

            self._memory_manager = MemoryManager(self.db_path)
        return self._memory_manager

    def get_gamma_client(self):
        """Get or create GammaMarketClient instance."""
        if self._gamma_client is None:
            from polymarket_agents.connectors.gamma import GammaMarketClient

            self._gamma_client = GammaMarketClient()
        return self._gamma_client

    def get_polymarket_client(self):
        """Get or create Polymarket client instance."""
        if self._polymarket_client is None:
            from polymarket_agents.connectors.polymarket import Polymarket

            self._polymarket_client = Polymarket()
        return self._polymarket_client

    def reset(self):
        """Reset cached instances. Useful for testing."""
        self._memory_manager = None
        self._gamma_client = None
        self._polymarket_client = None


# Global context (singleton pattern with override capability)
_context: Optional[AppContext] = None


def get_context() -> AppContext:
    """
    Get the current application context.

    Creates default context if none set.
    """
    global _context
    if _context is None:
        _context = AppContext()
    return _context


def set_context(ctx: AppContext) -> None:
    """
    Set the application context.

    Call at startup to configure dependencies.
    """
    global _context
    _context = ctx


def reset_context() -> None:
    """Reset to default context. Useful for testing."""
    global _context
    _context = None


# Convenience function for creating context from environment
def context_from_env() -> AppContext:
    """Create context from environment variables."""
    return AppContext(
        db_path=os.getenv("DATABASE_PATH", "data/markets.db"),
        default_model=os.getenv("DEFAULT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )

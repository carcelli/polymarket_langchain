"""
Domain registry for pluggable domain agents.

Domains register themselves here. LangChain tools discover them dynamically.

Usage:
    # Register a domain (done once at module load)
    from polymarket_agents.domains.registry import register_domain, DomainConfig

    register_domain(DomainConfig(
        name="crypto",
        description="Crypto binary price prediction markets",
        agent_factory=lambda ctx: CryptoAgent(price_source=ctx.price_source),
        scanner_factory=lambda ctx: CryptoScanner(price_source=ctx.price_source),
    ))

    # Get registered domains (from tools/agents)
    from polymarket_agents.domains.registry import get_domain, list_domains

    crypto = get_domain("crypto")
    agent = crypto.create_agent(context)
"""

from dataclasses import dataclass, field
from typing import Callable, Optional, Any, Protocol, TypeVar


class DataContext(Protocol):
    """
    Protocol for external data sources.

    Implement this to provide domain-specific data.
    Each domain uses the attributes it needs.
    """

    # Crypto
    price_source: Optional[Any]

    # NBA
    sports_source: Optional[Any]

    # Shared
    db_path: str


@dataclass
class DefaultContext:
    """Default context with sensible defaults."""

    price_source: Optional[Any] = None
    sports_source: Optional[Any] = None
    db_path: str = "data/markets.db"


T = TypeVar("T")


@dataclass
class DomainConfig:
    """
    Configuration for a registered domain.

    Attributes:
        name: Unique domain identifier (e.g., "crypto", "nba")
        description: Human-readable description for tool docs
        agent_factory: Creates domain agent from context
        scanner_factory: Creates domain scanner from context
        categories: Polymarket categories this domain handles
        tags: Additional metadata tags
    """

    name: str
    description: str
    agent_factory: Callable[[DataContext], Any]
    scanner_factory: Callable[[DataContext], Any]
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def create_agent(self, context: Optional[DataContext] = None):
        """Create agent instance with given context."""
        ctx = context or DefaultContext()
        return self.agent_factory(ctx)

    def create_scanner(self, context: Optional[DataContext] = None):
        """Create scanner instance with given context."""
        ctx = context or DefaultContext()
        return self.scanner_factory(ctx)


# Global registry
_DOMAINS: dict[str, DomainConfig] = {}


def register_domain(config: DomainConfig) -> None:
    """
    Register a domain.

    Called by domain modules to make themselves available.
    Idempotent - safe to call multiple times.
    """
    _DOMAINS[config.name] = config


def unregister_domain(name: str) -> None:
    """Remove a domain from registry."""
    _DOMAINS.pop(name, None)


def get_domain(name: str) -> Optional[DomainConfig]:
    """Get domain config by name."""
    return _DOMAINS.get(name)


def list_domains() -> list[str]:
    """List all registered domain names."""
    return list(_DOMAINS.keys())


def get_all_domains() -> dict[str, DomainConfig]:
    """Get all registered domains."""
    return _DOMAINS.copy()


def get_domains_by_category(category: str) -> list[DomainConfig]:
    """Find domains that handle a given category."""
    return [d for d in _DOMAINS.values() if category in d.categories]


def clear_registry() -> None:
    """Clear all registered domains. Useful for testing."""
    _DOMAINS.clear()


# Auto-registration helpers


def _register_builtin_domains() -> None:
    """Register built-in domains. Called on module import."""

    # Crypto domain
    try:
        from polymarket_agents.domains.crypto import CryptoAgent, CryptoScanner

        register_domain(
            DomainConfig(
                name="crypto",
                description="Scan and analyze crypto binary price prediction markets (BTC, ETH above $X)",
                agent_factory=lambda ctx: CryptoAgent(price_source=ctx.price_source),
                scanner_factory=lambda ctx: CryptoScanner(
                    price_source=ctx.price_source
                ),
                categories=["crypto"],
                tags=["bitcoin", "ethereum", "price", "binary"],
            )
        )
    except ImportError:
        pass  # Domain not available

    # NBA domain
    try:
        from polymarket_agents.domains.nba import NBAAgent, NBAScanner

        register_domain(
            DomainConfig(
                name="nba",
                description="Scan and analyze NBA game outcomes and player props with Log5 edge calculation",
                agent_factory=lambda ctx: NBAAgent(data_source=ctx.sports_source),
                scanner_factory=lambda ctx: NBAScanner(data_source=ctx.sports_source),
                categories=["sports"],
                tags=["nba", "basketball", "games", "props"],
            )
        )
    except ImportError:
        pass  # Domain not available


# Register on import
_register_builtin_domains()

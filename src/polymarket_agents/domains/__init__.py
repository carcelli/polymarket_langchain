"""
Domain-specific event scanners and agents.

Each domain (crypto, nba) is isolated with its own:
- Scanner: Fetches and filters relevant events from Polymarket
- Models: Domain-specific data structures
- Agent: Orchestrates scan -> enrich -> edge -> recommend
- Data sources: Protocol for external containers

Usage:
    # Crypto binary price predictions
    from polymarket_agents.domains.crypto import CryptoAgent
    crypto = CryptoAgent(price_source=my_price_container)
    crypto_recs = crypto.run()

    # NBA games and props
    from polymarket_agents.domains.nba import NBAAgent
    nba = NBAAgent(data_source=my_sports_container)
    nba_recs = nba.run()

    # Get as LangChain tools
    from polymarket_agents.langchain.domain_tools import get_crypto_tools, get_nba_tools

Implement PriceDataSource (crypto) or SportsDataSource (nba)
to connect your external data containers.

Register new domains:
    from polymarket_agents.domains.registry import register_domain, DomainConfig
"""

from .base import Edge, EventScanner, Market, ScanResult
from .registry import (
    DataContext,
    DefaultContext,
    DomainConfig,
    get_domain,
    list_domains,
    register_domain,
)

__all__ = [
    "Edge",
    "EventScanner",
    "Market",
    "ScanResult",
    "DataContext",
    "DefaultContext",
    "DomainConfig",
    "get_domain",
    "list_domains",
    "register_domain",
]

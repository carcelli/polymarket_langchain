"""
LangChain tool wrappers for domain agents.

Bridges the domain layer to LangChain tools.
Each domain agent becomes a callable tool.

Usage:
    from polymarket_agents.langchain.domain_tools import (
        get_crypto_tools,
        get_nba_tools,
        get_domain_tools,
    )

    # Get tools for specific domain
    crypto_tools = get_crypto_tools()

    # Get tools for any registered domain
    tools = get_domain_tools("nba")

    # Use with LangChain agent
    agent = create_react_agent(llm, tools=crypto_tools)
"""

from pydantic import BaseModel, Field

from polymarket_agents.tooling import wrap_tool
from polymarket_agents.domains.registry import (
    get_domain,
    list_domains as registry_list_domains,
    get_all_domains,
    DataContext,
    DefaultContext,
)


# Shared context for all domain tools (can be configured)
_context: DataContext = DefaultContext()


def set_context(context: DataContext) -> None:
    """Set the data context for all domain tools."""
    global _context
    _context = context


def get_context() -> DataContext:
    """Get current data context."""
    return _context


# --- Tool Input Schemas ---


class ScanMarketsInput(BaseModel):
    """Input for scanning markets."""

    min_volume: float = Field(default=5000, description="Minimum volume in USD")
    min_edge: float = Field(default=0.05, description="Minimum edge (0.05 = 5%)")
    max_results: int = Field(default=5, description="Maximum recommendations to return")


class ScanAssetInput(BaseModel):
    """Input for scanning specific crypto asset."""

    asset: str = Field(description="Asset symbol: BTC, ETH, SOL, XRP, DOGE")
    min_volume: float = Field(default=5000, description="Minimum volume in USD")
    min_edge: float = Field(default=0.05, description="Minimum edge")


class AnalyzeMatchupInput(BaseModel):
    """Input for NBA matchup analysis."""

    home_team: str = Field(description="Home team name (e.g., Lakers, Celtics)")
    away_team: str = Field(description="Away team name")


class ScanGamesInput(BaseModel):
    """Input for scanning NBA games."""

    min_volume: float = Field(default=5000, description="Minimum volume")
    min_edge: float = Field(default=0.05, description="Minimum edge")
    max_results: int = Field(default=5, description="Max recommendations")


class ScanPropsInput(BaseModel):
    """Input for scanning player props."""

    min_volume: float = Field(default=5000, description="Minimum volume")
    min_edge: float = Field(default=0.05, description="Minimum edge")
    max_results: int = Field(default=5, description="Max recommendations")


# --- Tool Implementations ---


def _format_recommendations(recommendations: list) -> str:
    """Format recommendations as readable string."""
    if not recommendations:
        return "No recommendations found matching criteria."

    lines = []
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"[{i}] {rec.market.question}")
        lines.append(f"    Action: {rec.action}")
        lines.append(f"    Edge: {rec.edge.edge:+.1%}")
        lines.append(f"    Kelly fraction: {rec.size_fraction:.1%}")
        lines.append(f"    Reasoning: {rec.reasoning}")
        lines.append("")

    return "\n".join(lines)


def _crypto_scan_impl(
    min_volume: float = 5000,
    min_edge: float = 0.05,
    max_results: int = 5,
) -> str:
    """Scan crypto binary price prediction markets for opportunities."""
    domain = get_domain("crypto")
    if not domain:
        return "Crypto domain not registered."

    agent = domain.create_agent(_context)
    agent.min_volume = min_volume
    agent.min_edge = min_edge
    agent.max_recommendations = max_results

    recommendations = agent.run()
    return _format_recommendations(recommendations)


def _crypto_scan_asset_impl(
    asset: str,
    min_volume: float = 5000,
    min_edge: float = 0.05,
) -> str:
    """Scan crypto markets for a specific asset (BTC, ETH, etc)."""
    domain = get_domain("crypto")
    if not domain:
        return "Crypto domain not registered."

    agent = domain.create_agent(_context)
    agent.min_volume = min_volume
    agent.min_edge = min_edge

    recommendations = agent.scan_asset(asset)
    return _format_recommendations(recommendations)


def _nba_scan_impl(
    min_volume: float = 5000,
    min_edge: float = 0.05,
    max_results: int = 5,
) -> str:
    """Scan all NBA markets (games and props) for betting opportunities."""
    domain = get_domain("nba")
    if not domain:
        return "NBA domain not registered."

    agent = domain.create_agent(_context)
    agent.min_volume = min_volume
    agent.min_edge = min_edge
    agent.max_recommendations = max_results

    recommendations = agent.run()
    return _format_recommendations(recommendations)


def _nba_scan_games_impl(
    min_volume: float = 5000,
    min_edge: float = 0.05,
    max_results: int = 5,
) -> str:
    """Scan NBA game outcome markets only (no player props)."""
    domain = get_domain("nba")
    if not domain:
        return "NBA domain not registered."

    agent = domain.create_agent(_context)
    agent.min_volume = min_volume
    agent.min_edge = min_edge
    agent.max_recommendations = max_results

    recommendations = agent.scan_games()
    return _format_recommendations(recommendations)


def _nba_scan_props_impl(
    min_volume: float = 5000,
    min_edge: float = 0.05,
    max_results: int = 5,
) -> str:
    """Scan NBA player prop markets only."""
    domain = get_domain("nba")
    if not domain:
        return "NBA domain not registered."

    agent = domain.create_agent(_context)
    agent.min_volume = min_volume
    agent.min_edge = min_edge
    agent.max_recommendations = max_results

    recommendations = agent.scan_props()
    return _format_recommendations(recommendations)


def _nba_analyze_matchup_impl(home_team: str, away_team: str) -> str:
    """Analyze NBA matchup using Log5 formula without looking for market."""
    domain = get_domain("nba")
    if not domain:
        return "NBA domain not registered."

    agent = domain.create_agent(_context)
    analysis = agent.analyze_matchup(home_team, away_team)

    if not analysis:
        return f"Could not analyze {home_team} vs {away_team}"

    lines = [
        f"Matchup: {analysis['home_team']} vs {analysis['away_team']}",
        f"Records: {analysis['home_record']} vs {analysis['away_record']}",
        f"Win %: {analysis['home_win_pct']:.1%} vs {analysis['away_win_pct']:.1%}",
        f"Neutral court probability: {analysis['neutral_prob']:.1%}",
        f"Home court adjusted: {analysis['home_prob']:.1%} (home) / {analysis['away_prob']:.1%} (away)",
    ]

    if analysis.get("key_factors"):
        lines.append(f"Key factors: {', '.join(analysis['key_factors'])}")

    return "\n".join(lines)


def _list_domains_impl() -> str:
    """List all registered domains and their descriptions."""
    domains = get_all_domains()
    if not domains:
        return "No domains registered."

    lines = ["Registered domains:"]
    for name, config in domains.items():
        lines.append(f"  - {name}: {config.description}")
        if config.categories:
            lines.append(f"    Categories: {', '.join(config.categories)}")

    return "\n".join(lines)


# --- Wrapped Tools ---

crypto_scan = wrap_tool(
    _crypto_scan_impl,
    name="crypto_scan",
    description="Scan crypto binary price prediction markets for trading opportunities. Returns markets with edge based on price analysis.",
    args_schema=ScanMarketsInput,
)

crypto_scan_asset = wrap_tool(
    _crypto_scan_asset_impl,
    name="crypto_scan_asset",
    description="Scan crypto markets for a specific asset (BTC, ETH, SOL, XRP, DOGE). Use when user asks about specific cryptocurrency.",
    args_schema=ScanAssetInput,
)

nba_scan = wrap_tool(
    _nba_scan_impl,
    name="nba_scan",
    description="Scan all NBA markets (games and player props) for betting opportunities with edge calculation.",
    args_schema=ScanMarketsInput,
)

nba_scan_games = wrap_tool(
    _nba_scan_games_impl,
    name="nba_scan_games",
    description="Scan NBA game outcome markets only (moneyline). Uses Log5 formula for edge calculation.",
    args_schema=ScanGamesInput,
)

nba_scan_props = wrap_tool(
    _nba_scan_props_impl,
    name="nba_scan_props",
    description="Scan NBA player prop markets (points, assists, rebounds).",
    args_schema=ScanPropsInput,
)

nba_analyze_matchup = wrap_tool(
    _nba_analyze_matchup_impl,
    name="nba_analyze_matchup",
    description="Analyze NBA matchup probability using Log5 formula. Use for quick matchup analysis without market lookup.",
    args_schema=AnalyzeMatchupInput,
)

list_domains_tool = wrap_tool(
    _list_domains_impl,
    name="list_domains",
    description="List all registered prediction market domains and their capabilities.",
)


# --- Tool Getters ---


def get_crypto_tools() -> list:
    """Get all crypto domain tools."""
    return [crypto_scan, crypto_scan_asset]


def get_nba_tools() -> list:
    """Get all NBA domain tools."""
    return [nba_scan, nba_scan_games, nba_scan_props, nba_analyze_matchup]


def get_domain_tools(domain_name: str) -> list:
    """
    Get tools for a specific domain by name.

    Args:
        domain_name: "crypto", "nba", or any registered domain

    Returns:
        List of LangChain tools for that domain
    """
    if domain_name == "crypto":
        return get_crypto_tools()
    elif domain_name == "nba":
        return get_nba_tools()

    # For dynamically registered domains, create generic scan tool
    domain = get_domain(domain_name)
    if not domain:
        return []

    def scan_impl(
        min_volume: float = 5000, min_edge: float = 0.05, max_results: int = 5
    ) -> str:
        agent = domain.create_agent(_context)
        if hasattr(agent, "min_volume"):
            agent.min_volume = min_volume
        if hasattr(agent, "min_edge"):
            agent.min_edge = min_edge
        if hasattr(agent, "max_recommendations"):
            agent.max_recommendations = max_results
        recommendations = agent.run()
        return _format_recommendations(recommendations)

    scan_tool = wrap_tool(
        scan_impl,
        name=f"{domain_name}_scan",
        description=domain.description,
        args_schema=ScanMarketsInput,
    )

    return [scan_tool]


def get_all_domain_tools() -> list:
    """Get tools for all registered domains."""
    tools = [list_domains_tool]
    for domain_name in registry_list_domains():
        tools.extend(get_domain_tools(domain_name))
    return tools

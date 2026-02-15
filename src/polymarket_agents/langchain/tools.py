"""
LangChain Tools for Polymarket API Integration

This module wraps all Polymarket agent functionality as LangChain tools.
Each tool has:
- Type-safe arguments using Pydantic
- Comprehensive docstrings (used by LLM to understand tool purpose)
- Error handling with informative messages

Tool implementations are defined as plain functions and wrapped via wrap_tool
to keep compatibility across different runtime configurations.

ARGUMENT REFERENCE
==================

LangChain tools use the @tool decorator which:
1. Parses the function docstring as the tool description
2. Infers argument types from type hints
3. Uses Pydantic models for complex argument validation

For more control, use StructuredTool with explicit args_schema.

Tool Argument Types:
- Simple types: str, int, float, bool
- Lists: List[str], List[int]
- Optional: Optional[str] = None
- Pydantic models for complex inputs

Example:
    @tool
    def my_tool(query: str, limit: int = 10) -> str:
        '''Tool description used by LLM.

        Args:
            query: Search query string
            limit: Maximum results to return
        '''
        return result
"""

import json
from typing import List, Optional, Dict, Callable

from pydantic import BaseModel, Field

from polymarket_agents.tooling import wrap_tool
from polymarket_agents.config import MARKET_FOCUS
from polymarket_agents.tools.research_tools import _fetch_documentation_impl

# Lazy imports to avoid circular dependencies
_polymarket = None
_gamma = None
_news = None
_chroma = None
_executor = None
_memory = None
_gamma_tool = None


def _get_memory():
    """Get the MemoryManager instance for database access."""
    global _memory
    if _memory is None:
        from polymarket_agents.memory.manager import MemoryManager

        _memory = MemoryManager("data/markets.db")
    return _memory


def _get_polymarket():
    global _polymarket
    if _polymarket is None:
        from polymarket_agents.connectors.polymarket import Polymarket

        _polymarket = Polymarket()
    return _polymarket


def _get_gamma():
    global _gamma
    if _gamma is None:
        from polymarket_agents.connectors.gamma import GammaMarketClient

        _gamma = GammaMarketClient()
    return _gamma


def _get_news():
    global _news
    if _news is None:
        from polymarket_agents.connectors.news import News

        _news = News()
    return _news


def _get_chroma():
    global _chroma
    if _chroma is None:
        from polymarket_agents.connectors.chroma import PolymarketRAG

        _chroma = PolymarketRAG()
    return _chroma


def _get_executor():
    global _executor
    if _executor is None:
        from polymarket_agents.application.executor import Executor

        _executor = Executor()
    return _executor


def _get_gamma_tool():
    """Get the Gamma markets tool instance."""
    global _gamma_tool
    if _gamma_tool is None:
        from polymarket_agents.tools.gamma_markets import GammaMarketsTool

        _gamma_tool = GammaMarketsTool()
    return _gamma_tool


# =============================================================================
# PYDANTIC SCHEMAS FOR COMPLEX TOOL INPUTS
# =============================================================================


class MarketQueryInput(BaseModel):
    """Schema for market query parameters."""

    limit: int = Field(default=10, description="Maximum number of markets to return")
    active_only: bool = Field(default=True, description="Only return active markets")


class EventQueryInput(BaseModel):
    """Schema for event query parameters."""

    limit: int = Field(default=10, description="Maximum number of events to return")
    tradeable_only: bool = Field(
        default=True, description="Only return tradeable events"
    )


class OrderInput(BaseModel):
    """Schema for trade order parameters."""

    token_id: str = Field(description="The CLOB token ID for the market outcome")
    price: float = Field(description="Price per share (0.01 to 0.99)")
    size: float = Field(description="Number of shares to trade")
    side: str = Field(description="'BUY' or 'SELL'")


class NewsQueryInput(BaseModel):
    """Schema for news search parameters."""

    keywords: str = Field(description="Comma-separated keywords to search for")
    date_start: Optional[str] = Field(
        default=None, description="Start date (YYYY-MM-DD)"
    )
    date_end: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")


class ForecastInput(BaseModel):
    """Schema for superforecaster analysis."""

    event_title: str = Field(description="Title of the prediction market event")
    market_question: str = Field(description="The specific question being predicted")
    outcome: str = Field(description="The outcome to analyze (e.g., 'Yes' or 'No')")


class TopVolumeMarketsInput(BaseModel):
    """Schema for getting top volume markets."""

    limit: int = Field(default=10, description="Maximum number of markets to return")
    category: Optional[str] = Field(
        default=None,
        description="Optional category filter (e.g., 'sports', 'politics')",
    )


class SearchMarketsInput(BaseModel):
    """Schema for searching markets in the database."""

    query: str = Field(description="Text to search for in market questions")
    limit: int = Field(default=10, description="Maximum number of results")
    category: Optional[str] = Field(
        default=None, description="Optional category filter"
    )


class MarketsByCategoryInput(BaseModel):
    """Schema for getting markets by category."""

    category: str = Field(
        description="Category to filter by (e.g., 'sports', 'politics', 'crypto')"
    )
    limit: int = Field(default=10, description="Maximum number of markets to return")


# =============================================================================
# MARKET TOOLS
# =============================================================================


def _fetch_all_markets_impl(limit: int = 20) -> str:
    """Fetch all available Polymarket prediction markets.

    Use this tool when you need to see what markets are available for trading.
    Returns a list of markets with their questions, prices, and trading status.

    Args:
        limit: Maximum number of markets to return (default: 20)

    Returns:
        JSON string containing list of markets with:
        - id: Market identifier
        - question: The prediction question
        - outcomes: Available outcomes (e.g., Yes/No)
        - outcome_prices: Current prices for each outcome
        - active: Whether market is currently tradeable
        - spread: Current bid-ask spread
    """
    try:
        poly = _get_polymarket()
        markets = poly.get_all_markets()[:limit]
        return json.dumps(
            [
                {
                    "id": m.id,
                    "question": m.question,
                    "outcomes": m.outcomes,
                    "outcome_prices": m.outcome_prices,
                    "active": m.active,
                    "spread": m.spread,
                    "description": (
                        m.description[:200] + "..."
                        if len(m.description) > 200
                        else m.description
                    ),
                }
                for m in markets
            ],
            indent=2,
        )
    except Exception as e:
        return f"Error fetching markets: {str(e)}"


def _fetch_tradeable_markets_impl(limit: int = 20) -> str:
    """Fetch only active, tradeable Polymarket markets.

    Use this tool when you specifically want markets that can be traded right now.
    Filters out closed, archived, or inactive markets.

    Args:
        limit: Maximum number of markets to return (default: 20)

    Returns:
        JSON string of tradeable markets with pricing info
    """
    try:
        poly = _get_polymarket()
        all_markets = poly.get_all_markets()[: limit * 2]  # Get extra to filter
        tradeable = poly.filter_markets_for_trading(all_markets)[:limit]
        return json.dumps(
            [
                {
                    "id": m.id,
                    "question": m.question,
                    "outcomes": m.outcomes,
                    "outcome_prices": m.outcome_prices,
                    "spread": m.spread,
                }
                for m in tradeable
            ],
            indent=2,
        )
    except Exception as e:
        return f"Error fetching tradeable markets: {str(e)}"


def _get_market_by_token_impl(token_id: str) -> str:
    """Get detailed information about a specific market by its token ID.

    Use this when you have a specific token ID and need full market details
    including description, end date, and current pricing.

    Args:
        token_id: The CLOB token ID (long numeric string)

    Returns:
        JSON string with complete market details
    """
    try:
        poly = _get_polymarket()
        market = poly.get_market(token_id)
        if market:
            return json.dumps(market, indent=2)
        return f"No market found for token_id: {token_id}"
    except Exception as e:
        return f"Error fetching market: {str(e)}"


def _get_market_by_id_impl(market_id: int) -> str:
    """Get detailed information about a market by its Gamma market ID.

    Use this when you have the numeric market ID from Gamma (not a CLOB token).

    Args:
        market_id: The Gamma market ID (integer)

    Returns:
        JSON string with complete market details from Gamma
    """
    try:
        gamma = _get_gamma()
        market = gamma.get_market(market_id)
        if market:
            return json.dumps(market, indent=2)
        return f"No market found for market_id: {market_id}"
    except Exception as e:
        return f"Error fetching market by id: {str(e)}"


def _get_current_markets_gamma_impl(limit: int = 10) -> str:
    """Fetch current active markets from Gamma API with extended data.

    The Gamma API provides additional market metadata not available
    from the main API, including liquidity and volume information.

    Args:
        limit: Maximum number of markets to return (default: 10)

    Returns:
        JSON string of markets with extended Gamma API data
    """
    try:
        gamma = _get_gamma()
        markets = gamma.get_current_markets(limit=limit)
        return json.dumps(markets, indent=2)
    except Exception as e:
        return f"Error fetching Gamma markets: {str(e)}"


def _get_clob_tradable_markets_impl(limit: int = 10) -> str:
    """Fetch markets with active CLOB (Central Limit Order Book) trading.

    These are markets where you can place limit orders at specific prices
    rather than just market orders.

    Args:
        limit: Maximum number of markets to return (default: 10)

    Returns:
        JSON string of CLOB-enabled markets
    """
    try:
        gamma = _get_gamma()
        markets = gamma.get_clob_tradable_markets(limit=limit)
        return json.dumps(markets, indent=2)
    except Exception as e:
        return f"Error fetching CLOB markets: {str(e)}"


# =============================================================================
# EVENT TOOLS
# =============================================================================


def _fetch_all_events_impl(limit: int = 20) -> str:
    """Fetch all Polymarket events (event = collection of related markets).

    Events group related markets together, e.g., "2024 US Election" event
    might contain markets for each candidate. Use this to understand
    the broader context of markets.

    Args:
        limit: Maximum number of events to return (default: 20)

    Returns:
        JSON string of events with their titles and associated market IDs
    """
    try:
        poly = _get_polymarket()
        events = poly.get_all_events()[:limit]
        return json.dumps(
            [
                {
                    "id": e.id,
                    "title": e.title,
                    "ticker": e.ticker,
                    "description": (
                        e.description[:200] + "..."
                        if len(e.description) > 200
                        else e.description
                    ),
                    "active": e.active,
                    "markets": e.markets,  # Comma-separated market IDs
                }
                for e in events
            ],
            indent=2,
        )
    except Exception as e:
        return f"Error fetching events: {str(e)}"


def _fetch_tradeable_events_impl(limit: int = 20) -> str:
    """Fetch only tradeable events (active, not restricted or archived).

    Filters events to only those where trading is currently possible.

    Args:
        limit: Maximum number of events to return (default: 20)

    Returns:
        JSON string of tradeable events
    """
    try:
        poly = _get_polymarket()
        events = poly.get_all_tradeable_events()[:limit]
        return json.dumps(
            [
                {
                    "id": e.id,
                    "title": e.title,
                    "ticker": e.ticker,
                    "markets": e.markets,
                }
                for e in events
            ],
            indent=2,
        )
    except Exception as e:
        return f"Error fetching tradeable events: {str(e)}"


def _get_event_by_id_impl(event_id: int) -> str:
    """Get detailed information about an event by its Gamma event ID.

    Use this when you have the numeric event ID from Gamma.

    Args:
        event_id: The Gamma event ID (integer)

    Returns:
        JSON string with complete event details from Gamma
    """
    try:
        gamma = _get_gamma()
        event = gamma.get_event(event_id)
        if event:
            return json.dumps(event, indent=2)
        return f"No event found for event_id: {event_id}"
    except Exception as e:
        return f"Error fetching event by id: {str(e)}"


def _get_current_events_gamma_impl(limit: int = 10) -> str:
    """Fetch current events from Gamma API with extended metadata.

    Args:
        limit: Maximum number of events to return (default: 10)

    Returns:
        JSON string of events with Gamma API data
    """
    try:
        gamma = _get_gamma()
        events = gamma.get_current_events(limit=limit)
        return json.dumps(events, indent=2)
    except Exception as e:
        return f"Error fetching Gamma events: {str(e)}"


# =============================================================================
# TAG TOOLS
# =============================================================================


def _fetch_all_tags_impl(limit: int = 20) -> str:
    """Fetch Polymarket tags.

    Tags are used to categorize events and markets (e.g., "Politics", "Sports").

    Args:
        limit: Maximum number of tags to return (default: 20)

    Returns:
        JSON string containing list of tags
    """
    try:
        gamma = _get_gamma()
        tags = gamma.get_tags(querystring_params={"limit": limit}, parse_pydantic=True)
        return json.dumps(
            [
                {
                    "id": t.id,
                    "label": t.label,
                    "slug": t.slug,
                }
                for t in tags
            ],
            indent=2,
        )
    except Exception as e:
        return f"Error fetching tags: {str(e)}"


def _get_tag_by_id_impl(tag_id: int) -> str:
    """Get detailed information about a tag by its ID.

    Args:
        tag_id: The tag ID (integer)

    Returns:
        JSON string with tag details
    """
    try:
        gamma = _get_gamma()
        tag = gamma.get_tag(tag_id)
        if tag:
            return json.dumps(tag, indent=2)
        return f"No tag found for tag_id: {tag_id}"
    except Exception as e:
        return f"Error fetching tag by id: {str(e)}"


# =============================================================================
# ORDER BOOK TOOLS
# =============================================================================


def _get_orderbook_impl(token_id: str) -> str:
    """Get the current order book for a market token.

    Shows all open buy (bid) and sell (ask) orders with their
    prices and sizes. Use this to understand market depth and
    find good entry/exit prices.

    Args:
        token_id: The CLOB token ID for the market outcome

    Returns:
        JSON string with bids and asks arrays containing price/size pairs
    """
    try:
        poly = _get_polymarket()
        if not poly.client:
            return "Error: Polymarket client not initialized. Check POLYGON_WALLET_PRIVATE_KEY."
        orderbook = poly.get_orderbook(token_id)
        return json.dumps(
            {
                "market": orderbook.market,
                "asset_id": orderbook.asset_id,
                "bids": [
                    {"price": b.price, "size": b.size} for b in orderbook.bids[:10]
                ],
                "asks": [
                    {"price": a.price, "size": a.size} for a in orderbook.asks[:10]
                ],
            },
            indent=2,
        )
    except Exception as e:
        return f"Error fetching orderbook: {str(e)}"


def _get_orderbook_price_impl(token_id: str) -> str:
    """Get the current mid-market price for a token.

    Args:
        token_id: The CLOB token ID

    Returns:
        Current price as a decimal (e.g., "0.65" = 65% probability)
    """
    try:
        poly = _get_polymarket()
        if not poly.client:
            return "Error: Polymarket client not initialized."
        price = poly.get_orderbook_price(token_id)
        return f"Current price: {price}"
    except Exception as e:
        return f"Error fetching price: {str(e)}"


# =============================================================================
# ACCOUNT & BALANCE TOOLS
# =============================================================================


def _get_usdc_balance_impl() -> str:
    """Get the current USDC balance of the configured wallet.

    This shows how much USDC is available for trading.
    Requires POLYGON_WALLET_PRIVATE_KEY to be configured.

    Returns:
        USDC balance as a string (e.g., "1234.56 USDC")
    """
    try:
        poly = _get_polymarket()
        balance = poly.get_usdc_balance()
        return f"USDC Balance: {balance:.2f}"
    except Exception as e:
        return f"Error fetching balance: {str(e)}"


def _get_wallet_address_impl() -> str:
    """Get the wallet address derived from the configured private key.

    Returns:
        The Polygon wallet address (0x...)
    """
    try:
        poly = _get_polymarket()
        address = poly.get_address_for_private_key()
        return f"Wallet Address: {address}"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# NEWS & RESEARCH TOOLS
# =============================================================================


def _search_news_impl(keywords: str) -> str:
    """Search for news articles related to given keywords.

    Use this to gather information for making predictions.
    Searches top headlines from NewsAPI.

    Args:
        keywords: Comma-separated keywords (e.g., "Biden,election,2024")

    Returns:
        JSON string of articles with title, description, and URL
    """
    try:
        news = _get_news()
        articles = news.get_articles_for_cli_keywords(keywords)
        return json.dumps(
            [
                {
                    "title": a.title,
                    "description": a.description,
                    "url": a.url,
                    "source": a.source.name if a.source else None,
                    "published_at": a.publishedAt,
                }
                for a in articles[:10]
            ],
            indent=2,
        )
    except Exception as e:
        return f"Error searching news: {str(e)}"


# =============================================================================
# ANALYSIS & FORECASTING TOOLS
# =============================================================================


def _get_superforecast_impl(
    event_title: str, market_question: str, outcome: str
) -> str:
    """Get a superforecaster-style probability estimate for a market outcome.

    Uses LLM with structured forecasting methodology to analyze the
    probability of a specific outcome. Implements Tetlock's superforecasting
    framework: base rates, factor decomposition, probabilistic reasoning.

    Args:
        event_title: Title/description of the event
        market_question: The specific prediction question
        outcome: The outcome to estimate (e.g., "Yes" or "No")

    Returns:
        Analysis with probability estimate and reasoning
    """
    try:
        executor = _get_executor()
        result = executor.get_superforecast(
            event_title=event_title, market_question=market_question, outcome=outcome
        )
        return result
    except Exception as e:
        return f"Error generating forecast: {str(e)}"


def _analyze_market_with_llm_impl(user_query: str) -> str:
    """Ask the LLM to analyze markets based on your query.

    Provides context about current Polymarket events and markets
    to the LLM for analysis. Good for general questions about
    what's happening in prediction markets.

    Args:
        user_query: Your question or analysis request

    Returns:
        LLM analysis based on current market data
    """
    try:
        executor = _get_executor()
        result = executor.get_polymarket_llm(user_query)
        return result
    except Exception as e:
        return f"Error analyzing: {str(e)}"


def _get_market_analyst_response_impl(question: str) -> str:
    """Get a market analyst response for a prediction question.

    Simpler than superforecast - just asks for a probability estimate.

    Args:
        question: The prediction question to analyze

    Returns:
        Probability estimate with brief reasoning
    """
    try:
        executor = _get_executor()
        result = executor.get_llm_response(question)
        return result
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# RAG (RETRIEVAL AUGMENTED GENERATION) TOOLS
# =============================================================================


def _query_markets_rag_impl(
    query: str, db_directory: str = "./local_db_markets"
) -> str:
    """Query the local RAG database of Polymarket markets.

    Uses vector similarity search to find markets relevant to your query.
    Must first create the RAG database using create_markets_rag_database.

    Args:
        query: Natural language query about markets
        db_directory: Path to the vector database directory

    Returns:
        Most relevant market documents with similarity scores
    """
    try:
        chroma = _get_chroma()
        results = chroma.query_local_markets_rag(
            local_directory=db_directory, query=query
        )
        return json.dumps(
            [
                {
                    "content": doc.page_content[:500],
                    "metadata": doc.metadata,
                    "score": score,
                }
                for doc, score in results[:5]
            ],
            indent=2,
        )
    except Exception as e:
        return f"Error querying RAG: {str(e)}"


def _create_markets_rag_database_impl(directory: str = "./local_db_markets") -> str:
    """Create a local vector database of current Polymarket markets.

    Downloads current markets and creates embeddings for similarity search.
    Run this periodically to keep the database updated.

    Args:
        directory: Where to store the vector database

    Returns:
        Confirmation message with number of markets indexed
    """
    try:
        chroma = _get_chroma()
        chroma.create_local_markets_rag(local_directory=directory)
        return f"Successfully created markets RAG database at {directory}"
    except Exception as e:
        return f"Error creating RAG database: {str(e)}"


# =============================================================================
# DATABASE / MEMORY TOOLS
# =============================================================================


def _get_database_stats_impl() -> str:
    """Get statistics about the markets database.

    Returns the total number of active markets, total volume, and category counts.
    Use this to understand what data is available before querying.

    Returns:
        Database statistics including market counts and volume by category
    """
    try:
        memory = _get_memory()
        stats = memory.get_stats()
        categories = memory.get_categories()

        result = {
            "total_active_markets": stats["total_markets"],
            "total_volume_usd": stats["total_volume"],
            "categories": [
                {
                    "name": cat["category"],
                    "market_count": cat["count"],
                    "volume_usd": cat["total_volume"],
                }
                for cat in categories
            ],
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting database stats: {str(e)}"


def _get_markets_by_category_impl(category: str = None, limit: int = 10) -> str:
    """Get markets from the database filtered by category.

    If MARKET_FOCUS environment variable is set and no category specified,
    defaults to that category. Set MARKET_FOCUS=sports to focus on sports.

    Categories include: politics, sports, crypto, tech, geopolitics,
    culture, finance, economy, science

    Args:
        category: Category name to filter by (e.g., 'sports', 'politics').
                 If not provided, uses MARKET_FOCUS environment variable.
        limit: Maximum number of markets to return (default: 10)

    Returns:
        List of markets in the specified category, sorted by volume
    """
    try:
        memory = _get_memory()
        # Use MARKET_FOCUS as default if no category specified
        effective_category = category or MARKET_FOCUS
        if not effective_category:
            return "Error: No category specified. Either provide a category parameter or set MARKET_FOCUS environment variable."
        markets = memory.list_markets_by_category(effective_category, limit=limit)

        result = []
        for m in markets:
            result.append(
                {
                    "id": m["id"],
                    "question": m["question"],
                    "category": m["category"],
                    "volume": m["volume"],
                    "liquidity": m["liquidity"],
                    "outcomes": m["outcomes"],
                    "outcome_prices": m["outcome_prices"],
                    "end_date": m["end_date"],
                    "slug": m.get("slug"),
                    "clob_token_ids": m.get("clob_token_ids"),
                }
            )

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting markets by category: {str(e)}"


def _get_top_volume_markets_impl(
    limit: int = 10, category: Optional[str] = None
) -> str:
    """Get the highest volume markets from the database.

    If MARKET_FOCUS environment variable is set, defaults to that category.
    Set MARKET_FOCUS=sports to focus on sports markets only.

    Args:
        limit: Maximum number of markets to return (default: 10)
        category: Optional category filter (e.g., 'sports', 'politics').
                 If not provided, uses MARKET_FOCUS environment variable.

    Returns:
        List of top markets by volume
    """
    try:
        memory = _get_memory()
        # Use MARKET_FOCUS as default if no category specified
        effective_category = category or MARKET_FOCUS
        markets = memory.list_top_volume_markets(
            limit=limit, category=effective_category
        )

        result = []
        for m in markets:
            result.append(
                {
                    "id": m["id"],
                    "question": m["question"],
                    "category": m["category"],
                    "volume": m["volume"],
                    "liquidity": m["liquidity"],
                    "outcomes": m["outcomes"],
                    "outcome_prices": m["outcome_prices"],
                    "end_date": m["end_date"],
                    "slug": m.get("slug"),
                }
            )

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error getting top volume markets: {str(e)}"


def _search_markets_db_impl(query: str, limit: int = 10, category: str = None) -> str:
    """Search markets in the database by question text.

    If MARKET_FOCUS environment variable is set, searches only within that category.
    Set MARKET_FOCUS=sports to focus on sports markets only.

    Performs a text search on market questions. For semantic search,
    use query_markets_rag instead.

    Args:
        query: Text to search for in market questions
        limit: Maximum number of results (default: 10)
        category: Optional category filter. If not provided, uses MARKET_FOCUS.

    Returns:
        List of matching markets sorted by volume
    """
    try:
        memory = _get_memory()
        # Use MARKET_FOCUS as default category filter if specified
        effective_category = category or MARKET_FOCUS
        markets = memory.search_markets(query, limit=limit, category=effective_category)

        result = []
        for m in markets:
            result.append(
                {
                    "id": m["id"],
                    "question": m["question"],
                    "category": m["category"],
                    "volume": m["volume"],
                    "outcomes": m["outcomes"],
                    "outcome_prices": m["outcome_prices"],
                    "end_date": m["end_date"],
                }
            )

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error searching markets: {str(e)}"


def _get_market_from_db_impl(market_id: str) -> str:
    """Get a specific market from the database by ID.

    Args:
        market_id: The market ID to look up

    Returns:
        Full market data including outcomes, prices, and metadata
    """
    try:
        memory = _get_memory()
        market = memory.get_market(market_id)

        if market:
            return json.dumps(market, indent=2)
        return f"Market {market_id} not found in database"
    except Exception as e:
        return f"Error getting market: {str(e)}"


def _list_recent_markets_impl(limit: int = 10) -> str:
    """List the most recently updated markets in the database.

    Args:
        limit: Maximum number of markets to return (default: 10)

    Returns:
        List of recently updated markets
    """
    try:
        memory = _get_memory()
        markets = memory.list_recent_markets(limit=limit)

        result = []
        for m in markets:
            result.append(
                {
                    "id": m["id"],
                    "question": m["question"],
                    "category": m["category"],
                    "volume": m["volume"],
                    "last_updated": m["last_updated"],
                    "outcomes": m["outcomes"],
                    "outcome_prices": m["outcome_prices"],
                }
            )

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error listing recent markets: {str(e)}"


# =============================================================================
# TRADING TOOLS (USE WITH CAUTION)
# =============================================================================


def _preview_order_impl(token_id: str, price: float, size: float, side: str) -> str:
    """Preview what an order would look like (does NOT execute).

    Use this to understand order parameters before placing real trades.

    Args:
        token_id: The CLOB token ID for the market outcome
        price: Price per share (0.01 to 0.99)
        size: Number of shares to trade
        side: 'BUY' or 'SELL'

    Returns:
        Preview of the order with estimated cost
    """
    try:
        side = side.upper()
        if side not in ["BUY", "SELL"]:
            return "Error: side must be 'BUY' or 'SELL'"
        if not 0.01 <= price <= 0.99:
            return "Error: price must be between 0.01 and 0.99"

        estimated_cost = price * size if side == "BUY" else 0
        potential_payout = size if side == "BUY" else price * size

        return json.dumps(
            {
                "preview": True,
                "token_id": token_id,
                "side": side,
                "price": price,
                "size": size,
                "estimated_cost_usdc": round(estimated_cost, 2),
                "potential_payout_usdc": round(potential_payout, 2),
                "max_profit_usdc": round(potential_payout - estimated_cost, 2),
                "warning": "This is a PREVIEW only. No order was placed.",
            },
            indent=2,
        )
    except Exception as e:
        return f"Error: {str(e)}"


# Note: Actual trading functions are commented out for safety
# Uncomment and use with extreme caution - real money at stake!

# @tool
# def execute_limit_order(token_id: str, price: float, size: float, side: str) -> str:
#     """Execute a limit order on Polymarket.
#
#     WARNING: This executes a REAL trade with REAL money.
#
#     Args:
#         token_id: The CLOB token ID
#         price: Limit price (0.01 to 0.99)
#         size: Number of shares
#         side: 'BUY' or 'SELL'
#     """
#     poly = _get_polymarket()
#     result = poly.execute_order(price, size, side, token_id)
#     return json.dumps(result, indent=2)


# =============================================================================
# TOOL WRAPPERS
# =============================================================================

fetch_all_markets = wrap_tool(_fetch_all_markets_impl, name="fetch_all_markets")
fetch_tradeable_markets = wrap_tool(
    _fetch_tradeable_markets_impl,
    name="fetch_tradeable_markets",
)
get_market_by_token = wrap_tool(_get_market_by_token_impl, name="get_market_by_token")
get_market_by_id = wrap_tool(_get_market_by_id_impl, name="get_market_by_id")
get_current_markets_gamma = wrap_tool(
    _get_current_markets_gamma_impl,
    name="get_current_markets_gamma",
)
get_clob_tradable_markets = wrap_tool(
    _get_clob_tradable_markets_impl,
    name="get_clob_tradable_markets",
)

fetch_all_events = wrap_tool(_fetch_all_events_impl, name="fetch_all_events")
fetch_tradeable_events = wrap_tool(
    _fetch_tradeable_events_impl,
    name="fetch_tradeable_events",
)
get_event_by_id = wrap_tool(_get_event_by_id_impl, name="get_event_by_id")
get_current_events_gamma = wrap_tool(
    _get_current_events_gamma_impl,
    name="get_current_events_gamma",
)

fetch_all_tags = wrap_tool(_fetch_all_tags_impl, name="fetch_all_tags")
get_tag_by_id = wrap_tool(_get_tag_by_id_impl, name="get_tag_by_id")

get_orderbook = wrap_tool(_get_orderbook_impl, name="get_orderbook")
get_orderbook_price = wrap_tool(
    _get_orderbook_price_impl,
    name="get_orderbook_price",
)

get_usdc_balance = wrap_tool(_get_usdc_balance_impl, name="get_usdc_balance")
get_wallet_address = wrap_tool(_get_wallet_address_impl, name="get_wallet_address")

search_news = wrap_tool(_search_news_impl, name="search_news")

get_superforecast = wrap_tool(_get_superforecast_impl, name="get_superforecast")
analyze_market_with_llm = wrap_tool(
    _analyze_market_with_llm_impl,
    name="analyze_market_with_llm",
)
get_market_analyst_response = wrap_tool(
    _get_market_analyst_response_impl,
    name="get_market_analyst_response",
)

query_markets_rag = wrap_tool(_query_markets_rag_impl, name="query_markets_rag")
create_markets_rag_database = wrap_tool(
    _create_markets_rag_database_impl,
    name="create_markets_rag_database",
)

# Database / Memory Tools
get_database_stats = wrap_tool(_get_database_stats_impl, name="get_database_stats")
get_markets_by_category = wrap_tool(
    _get_markets_by_category_impl,
    name="get_markets_by_category",
    args_schema=MarketsByCategoryInput,
)
get_top_volume_markets = wrap_tool(
    _get_top_volume_markets_impl,
    name="get_top_volume_markets",
    args_schema=TopVolumeMarketsInput,
)
search_markets_db = wrap_tool(
    _search_markets_db_impl,
    name="search_markets_db",
    args_schema=SearchMarketsInput,
)
get_market_from_db = wrap_tool(_get_market_from_db_impl, name="get_market_from_db")
list_recent_markets = wrap_tool(_list_recent_markets_impl, name="list_recent_markets")

preview_order = wrap_tool(_preview_order_impl, name="preview_order")

_TOOL_FUNCTIONS: Dict[str, Callable] = {
    "fetch_all_markets": _fetch_all_markets_impl,
    "fetch_tradeable_markets": _fetch_tradeable_markets_impl,
    "get_market_by_token": _get_market_by_token_impl,
    "get_market_by_id": _get_market_by_id_impl,
    "get_current_markets_gamma": _get_current_markets_gamma_impl,
    "get_clob_tradable_markets": _get_clob_tradable_markets_impl,
    "gamma_fetch_markets": lambda **kwargs: _get_gamma_tool()._run(**kwargs),
    "fetch_all_events": _fetch_all_events_impl,
    "fetch_tradeable_events": _fetch_tradeable_events_impl,
    "get_event_by_id": _get_event_by_id_impl,
    "get_current_events_gamma": _get_current_events_gamma_impl,
    "fetch_all_tags": _fetch_all_tags_impl,
    "get_tag_by_id": _get_tag_by_id_impl,
    "get_orderbook": _get_orderbook_impl,
    "get_orderbook_price": _get_orderbook_price_impl,
    "get_usdc_balance": _get_usdc_balance_impl,
    "get_wallet_address": _get_wallet_address_impl,
    "search_news": _search_news_impl,
    "get_superforecast": _get_superforecast_impl,
    "analyze_market_with_llm": _analyze_market_with_llm_impl,
    "get_market_analyst_response": _get_market_analyst_response_impl,
    "query_markets_rag": _query_markets_rag_impl,
    "create_markets_rag_database": _create_markets_rag_database_impl,
    "get_database_stats": _get_database_stats_impl,
    "get_markets_by_category": _get_markets_by_category_impl,
    "get_top_volume_markets": _get_top_volume_markets_impl,
    "search_markets_db": _search_markets_db_impl,
    "get_market_from_db": _get_market_from_db_impl,
    "list_recent_markets": _list_recent_markets_impl,
    "preview_order": _preview_order_impl,
}

_TOOL_FUNCTIONS["fetch_documentation"] = _fetch_documentation_impl

fetch_documentation = wrap_tool(_fetch_documentation_impl, name="fetch_documentation")


def get_tool_functions() -> Dict[str, Callable]:
    """Get raw tool callables for non-LangChain configurations."""
    return dict(_TOOL_FUNCTIONS)


# =============================================================================
# TOOL COLLECTION FUNCTIONS
# =============================================================================


def get_market_tools() -> List:
    """Get all market-related tools."""
    return [
        fetch_all_markets,
        fetch_tradeable_markets,
        get_market_by_token,
        get_market_by_id,
        get_current_markets_gamma,
        get_clob_tradable_markets,
        _get_gamma_tool(),  # Gamma API markets tool
    ]


def get_event_tools() -> List:
    """Get all event-related tools."""
    return [
        fetch_all_events,
        fetch_tradeable_events,
        get_event_by_id,
        get_current_events_gamma,
    ]


def get_tag_tools() -> List:
    """Get all tag-related tools."""
    return [
        fetch_all_tags,
        get_tag_by_id,
    ]


def get_read_only_tools() -> List:
    """Get read-only market and event tools (no trading or wallet access)."""
    return get_market_tools() + get_event_tools() + get_tag_tools()


def get_trading_tools() -> List:
    """Get trading and order book tools."""
    return [
        get_orderbook,
        get_orderbook_price,
        get_usdc_balance,
        get_wallet_address,
        preview_order,
    ]


def get_analysis_tools() -> List:
    """Get analysis and forecasting tools."""
    return [
        get_superforecast,
        analyze_market_with_llm,
        get_market_analyst_response,
        search_news,
        query_markets_rag,
        create_markets_rag_database,
        fetch_documentation,
    ]


# Database / Memory Tools


def get_database_tools() -> List:
    """Get database/memory tools for accessing stored market data."""
    return [
        get_database_stats,
        get_markets_by_category,
        get_top_volume_markets,
        search_markets_db,
        get_market_from_db,
        list_recent_markets,
    ]


# Domain Tools (crypto, nba, etc.)


def get_domain_tools(domain: str = None) -> List:
    """
    Get domain-specific tools (crypto, nba, etc).

    Args:
        domain: Specific domain ("crypto", "nba") or None for all.

    These tools wrap specialized domain agents that scan Polymarket
    for opportunities with edge calculation.
    """
    try:
        from polymarket_agents.langchain.domain_tools import (
            get_crypto_tools,
            get_nba_tools,
            get_all_domain_tools,
        )

        if domain == "crypto":
            return get_crypto_tools()
        elif domain == "nba":
            return get_nba_tools()
        elif domain is None:
            return get_all_domain_tools()
        else:
            # Try to get tools for unknown domain
            from polymarket_agents.langchain.domain_tools import (
                get_domain_tools as _get,
            )

            return _get(domain)
    except ImportError:
        return []


def get_all_tools(include_domains: bool = True) -> List:
    """
    Get all available Polymarket tools for LangChain agents.

    Args:
        include_domains: Include domain-specific tools (crypto, nba).
    """
    tools = (
        get_market_tools()
        + get_event_tools()
        + get_tag_tools()
        + get_trading_tools()
        + get_analysis_tools()
        + get_database_tools()
    )

    if include_domains:
        tools = tools + get_domain_tools()

    return tools


# =============================================================================
# ARGUMENT REFERENCE DOCUMENTATION
# =============================================================================

ARGUMENT_REFERENCE = """
POLYMARKET LANGCHAIN TOOLS - ARGUMENT REFERENCE
================================================

1. MARKET TOOLS
---------------

fetch_all_markets(limit: int = 20)
    limit: Max markets to return. Type: int. Default: 20

fetch_tradeable_markets(limit: int = 20)
    limit: Max markets to return. Type: int. Default: 20

get_market_by_token(token_id: str)
    token_id: CLOB token ID (long numeric string). Type: str. Required.
    Example: "101669189743438912873361127612589311253202068943959811456820079057046819967115"

get_market_by_id(market_id: int)
    market_id: Gamma market ID. Type: int. Required.

get_current_markets_gamma(limit: int = 10)
    limit: Max markets. Type: int. Default: 10

get_clob_tradable_markets(limit: int = 10)
    limit: Max markets. Type: int. Default: 10

gamma_fetch_markets(active: bool = True, limit: int = 50, question_contains: str = None)
    active: Only return open/active markets. Type: bool. Default: True
    limit: Max markets to return (1-100). Type: int. Default: 50
    question_contains: Filter by question substring. Type: str. Default: None


2. EVENT TOOLS
--------------

fetch_all_events(limit: int = 20)
    limit: Max events. Type: int. Default: 20

fetch_tradeable_events(limit: int = 20)
    limit: Max events. Type: int. Default: 20

get_event_by_id(event_id: int)
    event_id: Gamma event ID. Type: int. Required.

get_current_events_gamma(limit: int = 10)
    limit: Max events. Type: int. Default: 10


3. TAG TOOLS
------------

fetch_all_tags(limit: int = 20)
    limit: Max tags. Type: int. Default: 20

get_tag_by_id(tag_id: int)
    tag_id: Gamma tag ID. Type: int. Required.


4. ORDERBOOK TOOLS
------------------

get_orderbook(token_id: str)
    token_id: CLOB token ID. Type: str. Required.

get_orderbook_price(token_id: str)
    token_id: CLOB token ID. Type: str. Required.


4. ACCOUNT TOOLS
----------------

get_usdc_balance() -> No arguments

get_wallet_address() -> No arguments


5. NEWS & RESEARCH TOOLS
------------------------

search_news(keywords: str)
    keywords: Comma-separated search terms. Type: str. Required.
    Example: "Trump,Biden,election"


6. ANALYSIS TOOLS
-----------------

get_superforecast(event_title: str, market_question: str, outcome: str)
    event_title: Event description. Type: str. Required.
    market_question: The prediction question. Type: str. Required.
    outcome: Outcome to analyze (e.g., "Yes"). Type: str. Required.

analyze_market_with_llm(user_query: str)
    user_query: Natural language query. Type: str. Required.

get_market_analyst_response(question: str)
    question: Prediction question. Type: str. Required.


7. RAG TOOLS
------------

query_markets_rag(query: str, db_directory: str = "./local_db_markets")
    query: Search query. Type: str. Required.
    db_directory: Vector DB path. Type: str. Default: "./local_db_markets"

create_markets_rag_database(directory: str = "./local_db_markets")
    directory: Where to store DB. Type: str. Default: "./local_db_markets"


8. TRADING TOOLS
----------------

preview_order(token_id: str, price: float, size: float, side: str)
    token_id: CLOB token ID. Type: str. Required.
    price: 0.01 to 0.99. Type: float. Required.
    size: Number of shares. Type: float. Required.
    side: "BUY" or "SELL". Type: str. Required.


PYDANTIC SCHEMAS FOR STRUCTURED TOOLS
=====================================

For complex inputs, use StructuredTool with args_schema:

    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field

    class MyInput(BaseModel):
        query: str = Field(description="Search query")
        limit: int = Field(default=10, ge=1, le=100)

    tool = StructuredTool.from_function(
        func=my_function,
        name="my_tool",
        description="Tool description",
        args_schema=MyInput
    )
"""


def print_argument_reference():
    """Print the complete argument reference."""
    print(ARGUMENT_REFERENCE)

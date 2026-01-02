"""
Research Tools for Enhanced Polymarket Analysis

Integrates web search capabilities with existing market tools to provide
comprehensive research for trading decisions.
"""

import os
from typing import Literal, List, Dict, Any
from tavily import TavilyClient

from agents.tooling import wrap_tool
from agents.polymarket.gamma import GammaMarketClient

# Initialize clients
tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
gamma_client = GammaMarketClient()


def _web_search_impl(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
) -> Dict[str, Any]:
    """
    Search the web for information relevant to market analysis.

    Use this to gather external context, news, and research about markets,
    events, or entities mentioned in Polymarket questions.

    Args:
        query: Search query (e.g., "Trump 2028 election polls")
        max_results: Number of results to return (1-10)
        topic: Search topic category
        include_raw_content: Whether to include full article content

    Returns:
        Search results with titles, URLs, snippets, and optionally full content
    """
    try:
        results = tavily_client.search(
            query=query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
        )
        return results
    except Exception as e:
        return {"error": f"Web search failed: {str(e)}"}


def _market_news_search_impl(market_question: str, days_back: int = 7) -> List[Dict]:
    """
    Search for recent news specifically related to a market question.

    This combines market context with web search to find relevant news
    that could impact market prices.

    Args:
        market_question: The Polymarket question
        days_back: How many days of news to search

    Returns:
        List of news articles with relevance scores
    """
    try:
        # Extract key entities from the market question
        # This is a simple heuristic - could be enhanced with NLP
        question_lower = market_question.lower()

        # Look for common entities that might have news
        search_terms = []

        # Political figures
        politicians = ["trump", "biden", "harris", "desantis", "haley"]
        for pol in politicians:
            if pol in question_lower:
                search_terms.append(f"{pol} news")

        # Sports
        sports_terms = ["super bowl", "world series", "championship", "playoffs"]
        for term in sports_terms:
            if term in question_lower:
                search_terms.append(f"{term} latest")

        # Economic indicators
        econ_terms = ["fed", "interest rate", "inflation", "gdp", "unemployment"]
        for term in econ_terms:
            if term in question_lower:
                search_terms.append(f"{term} latest news")

        # If no specific terms found, use the whole question
        if not search_terms:
            search_terms = [market_question]

        # Search for news
        all_results = []
        for term in search_terms[:2]:  # Limit to 2 searches
            results = tavily_client.search(
                query=f"{term} {days_back}d",
                max_results=3,
                topic="news",
                include_raw_content=False,
            )

            if "results" in results:
                for result in results["results"]:
                    result["search_term"] = term
                    all_results.append(result)

        return all_results[:5]  # Return top 5 results

    except Exception as e:
        return [{"error": f"Market news search failed: {str(e)}"}]


def _comprehensive_research_impl(market_id: str) -> Dict[str, Any]:
    """
    Perform comprehensive research on a market including:
    - Market details from Polymarket
    - Related web search results
    - Recent news and developments

    Args:
        market_id: Polymarket market ID

    Returns:
        Comprehensive research package
    """
    try:
        research_package = {
            "market_id": market_id,
            "timestamp": "2025-01-01T00:00:00Z",  # Would be datetime.now()
            "market_data": {},
            "web_research": [],
            "news_context": [],
            "analysis_ready": False,
        }

        # Get market data
        try:
            markets = gamma_client.get_markets(
                querystring_params={"id": market_id}, parse_pydantic=True
            )
            if markets:
                research_package["market_data"] = markets[0].dict()
        except Exception as e:
            research_package["market_data"] = {
                "error": f"Failed to fetch market: {str(e)}"
            }

        # Get market question for research
        market_question = research_package.get("market_data", {}).get("question", "")

        if market_question:
            # Web search for general context
            web_results = _web_search_impl(
                query=f"{market_question} analysis", max_results=3, topic="general"
            )
            research_package["web_research"] = web_results.get("results", [])

            # News search
            news_results = _market_news_search_impl(market_question, days_back=7)
            research_package["news_context"] = news_results

            research_package["analysis_ready"] = True

        return research_package

    except Exception as e:
        return {
            "error": f"Comprehensive research failed: {str(e)}",
            "market_id": market_id,
        }


# Wrapped tools for use in agents
web_search = wrap_tool(
    _web_search_impl,
    name="web_search",
    description="Search the web for information relevant to market analysis",
)

market_news_search = wrap_tool(
    _market_news_search_impl,
    name="market_news_search",
    description="Search for recent news related to a market question",
)

comprehensive_research = wrap_tool(
    _comprehensive_research_impl,
    name="comprehensive_research",
    description="Perform comprehensive research on a market including data, web results, and news",
)





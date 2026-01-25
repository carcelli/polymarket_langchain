"""
Market Research Subagent

Specialized subagent for in-depth market research and analysis.
Handles complex research tasks that would bloat the main agent's context.
"""

import os
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Import our existing tools and utilities
from polymarket_agents.analysis import MarketAnalyzer
from polymarket_agents.utils.text import asciize
from polymarket_agents.ml_strategies.registry import best_strategy


def search_related_markets(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for markets related to a query.

    This tool searches our local market database for relevant markets
    without making external API calls.
    """
    analyzer = MarketAnalyzer()

    # Get markets by category or keyword matching
    all_markets = analyzer.get_high_volume_markets(limit=limit * 3)

    # Simple relevance scoring based on question text
    query_lower = query.lower()
    scored_markets = []

    for market in all_markets:
        question = market["question"].lower()
        score = 0

        # Exact word matches get higher scores
        query_words = set(query_lower.split())
        question_words = set(question.split())

        # Word overlap
        overlap = len(query_words.intersection(question_words))
        score += overlap * 2

        # Partial matches
        for q_word in query_words:
            if q_word in question:
                score += 1

        if score > 0:
            scored_markets.append((market, score))

    # Sort by relevance and return top results
    scored_markets.sort(key=lambda x: x[1], reverse=True)

    results = []
    for market, score in scored_markets[:limit]:
        results.append(
            {
                "market": market,
                "relevance_score": score,
                "category": market["category"],
                "volume": market["volume"],
                "probability": market["implied_probability"],
            }
        )

    return results


def analyze_market_with_strategies(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a market using the registered ML strategy system.

    Uses best_strategy to automatically select and run the most promising
    ML strategy for the given market data.
    """
    try:
        # Get ML strategy recommendation
        strategy_result = best_strategy(market_data)

        if "error" in strategy_result:
            return {
                "market_id": market_data.get("id", "unknown"),
                "analysis_type": "ml_strategy",
                "error": strategy_result["error"],
                "strategies_available": strategy_result.get("strategies_tried", 0),
            }

        # Format for research output
        return {
            "market_id": market_data.get("id", market_data.get("market_id", "unknown")),
            "market_question": market_data.get("question", "Unknown market"),
            "analysis_type": "ml_strategy",
            "selected_strategy": strategy_result.get("selected_strategy", "unknown"),
            "edge": strategy_result.get("edge", 0.0),
            "recommendation": strategy_result.get("recommendation", "HOLD"),
            "confidence": strategy_result.get("confidence", 0.0),
            "reasoning": strategy_result.get("reasoning", "ML strategy analysis"),
            "strategies_compared": strategy_result.get("strategies_compared", 0),
            "additional_insights": {
                "momentum": strategy_result.get("momentum"),
                "volume_ratio": strategy_result.get("volume_ratio"),
                "data_points": strategy_result.get("data_points"),
            },
        }

    except Exception as e:
        return {
            "market_id": market_data.get("id", "unknown"),
            "analysis_type": "ml_strategy",
            "error": f"Strategy analysis failed: {str(e)}",
        }


def analyze_market_trends(market_id: str, days_back: int = 30) -> Dict[str, Any]:
    """
    Analyze historical trends for a specific market.

    This would analyze price movements, volume changes, etc.
    For now, returns mock data based on current market state.
    """
    analyzer = MarketAnalyzer()

    # Get current market data
    markets = analyzer.get_high_volume_markets(limit=100)
    market_data = None

    for market in markets:
        if str(market["id"]) == str(market_id):
            market_data = market
            break

    if not market_data:
        return {"error": f"Market {market_id} not found"}

    # Mock trend analysis (in real implementation, this would use historical data)
    current_prob = market_data["implied_probability"]
    volume = market_data["volume"]

    # Simulate trend analysis
    trend_analysis = {
        "market_id": market_id,
        "current_probability": current_prob,
        "volume": volume,
        "trend_direction": "stable",  # Could be 'up', 'down', 'stable'
        "volatility": "medium",  # Could be 'low', 'medium', 'high'
        "momentum": 0.0,  # -1 to 1 scale
        "key_insights": [
            f"Market shows {current_prob:.1%} implied probability",
            f"Trading volume of ${volume:,.0f} indicates {'high' if volume > 10000000 else 'moderate'} interest",
            "Price action suggests market expectations are stabilizing",
        ],
        "recommendations": [
            "Monitor for significant volume changes",
            "Watch for news events that could shift probability",
            "Consider position sizing based on volatility assessment",
        ],
    }

    return trend_analysis


def create_market_research_subagent():
    """
    Create the market research subagent configuration.

    This subagent specializes in:
    - Finding related markets
    - Analyzing market trends
    - Researching market context
    - Providing detailed market intelligence
    """

    return {
        "name": "market-research",
        "description": "Conducts in-depth market research, finds related markets, analyzes trends, and provides detailed market intelligence. Use for complex research tasks that require multiple data sources and analysis.",
        "system_prompt": """You are a specialized market research analyst for prediction markets.

Your role is to provide comprehensive market intelligence that would otherwise clutter the main agent's context.

RESEARCH PROCESS:
1. Start with the user's query and break it down into research components
2. Use search_related_markets to find relevant markets in our database
3. Use analyze_market_trends to understand market dynamics
4. Synthesize findings into actionable insights

OUTPUT FORMAT:
Return your response in this exact structure:

MARKET OVERVIEW
- Key markets found: [list 2-3 most relevant]
- Market categories: [categories represented]
- Volume distribution: [summary of trading activity]

TREND ANALYSIS
- Price movements: [significant trends observed]
- Volume patterns: [unusual activity or changes]
- Market sentiment: [overall market expectations]

KEY INSIGHTS
- [Bullet points of important findings]
- [Market relationships or correlations]
- [Risk factors or opportunities identified]

RECOMMENDATIONS
- [Specific, actionable recommendations]
- [Markets to monitor closely]
- [Research follow-ups needed]

   Keep your response under 600 words. Focus on synthesis, not raw data.""",
        "tools": [
            search_related_markets,
            analyze_market_trends,
            analyze_market_with_strategies,
        ],
        # Use a more analytical model for research
        "model": "gpt-4o",  # Could be overridden based on available models
    }

"""
Data Collection Subagent

Specialized subagent for gathering and organizing market data from various sources.
Handles data collection tasks that would otherwise clutter the main agent's context.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

# Import our existing utilities
from market_analysis_workflow import MarketAnalyzer


def collect_market_data(query: str, categories: List[str] = None, min_volume: float = 100000,
                       max_results: int = 50) -> Dict[str, Any]:
    """
    Collect comprehensive market data based on search criteria.

    Args:
        query: Search query or keywords
        categories: List of categories to focus on
        min_volume: Minimum trading volume threshold
        max_results: Maximum number of markets to return
    """
    analyzer = MarketAnalyzer()

    # Get markets matching criteria
    markets = analyzer.get_high_volume_markets(
        category=categories[0] if categories else None,
        limit=max_results * 2,  # Get more to filter
        min_volume=min_volume
    )

    # Filter by query relevance if provided
    if query and query.lower() != "all":
        query_lower = query.lower()
        filtered_markets = []

        for market in markets:
            question = market['question'].lower()
            category = market.get('category', '').lower()

            # Check relevance
            relevance_score = 0

            # Exact keyword matches
            query_words = set(query_lower.split())
            question_words = set(question.split())

            overlap = len(query_words.intersection(question_words))
            relevance_score += overlap * 3

            # Partial matches in question
            for q_word in query_words:
                if q_word in question:
                    relevance_score += 1

            # Category matches
            if any(cat.lower() in category for cat in (categories or [])):
                relevance_score += 2

            if relevance_score > 0:
                market['relevance_score'] = relevance_score
                filtered_markets.append(market)

        # Sort by relevance and volume
        filtered_markets.sort(key=lambda x: (x.get('relevance_score', 0), x.get('volume', 0)), reverse=True)
        markets = filtered_markets[:max_results]
    else:
        markets = markets[:max_results]

    # Organize results
    data_collection = {
        "query": query,
        "categories_searched": categories or ["all"],
        "total_markets_found": len(markets),
        "min_volume_threshold": min_volume,

        "markets": markets,

        "summary_stats": {
            "total_volume": sum(m.get('volume', 0) for m in markets),
            "avg_volume": sum(m.get('volume', 0) for m in markets) / max(1, len(markets)),
            "categories_represented": list(set(m.get('category', 'unknown') for m in markets)),
            "volume_range": {
                "min": min((m.get('volume', 0) for m in markets), default=0),
                "max": max((m.get('volume', 0) for m in markets), default=0)
            }
        },

        "data_quality": {
            "complete_records": sum(1 for m in markets if all(k in m for k in ['id', 'question', 'volume', 'category'])),
            "has_price_data": sum(1 for m in markets if 'outcome_prices' in m),
            "has_volume_data": sum(1 for m in markets if m.get('volume', 0) > 0),
            "recent_data": sum(1 for m in markets if 'last_updated' in m)
        },

        "collection_metadata": {
            "timestamp": datetime.now().isoformat(),
            "data_source": "Polymarket API + Local Database",
            "collection_method": "Filtered search with volume thresholds"
        }
    }

    return data_collection


def gather_market_intelligence(market_ids: List[str], include_historical: bool = False) -> Dict[str, Any]:
    """
    Gather detailed intelligence on specific markets.

    Args:
        market_ids: List of market IDs to analyze
        include_historical: Whether to include historical data
    """
    analyzer = MarketAnalyzer()

    intelligence_data = {
        "market_ids_requested": market_ids,
        "markets_analyzed": [],
        "intelligence_summary": {},
        "timestamp": datetime.now().isoformat()
    }

    # Get all high-volume markets for context
    all_markets = analyzer.get_high_volume_markets(limit=1000)

    for market_id in market_ids:
        market_data = None

        # Find the specific market
        for market in all_markets:
            if str(market.get('id', '')) == str(market_id):
                market_data = market
                break

        if not market_data:
            intelligence_data["markets_analyzed"].append({
                "market_id": market_id,
                "status": "not_found",
                "error": f"Market {market_id} not found in database"
            })
            continue

        # Gather intelligence on this market
        market_intelligence = {
            "market_id": market_id,
            "status": "analyzed",
            "market_data": market_data,

            "market_context": {
                "category_rank": None,  # Would calculate position in category
                "volume_percentile": None,  # Would calculate volume ranking
                "similar_markets": []  # Would find related markets
            },

            "trading_signals": {
                "volume_trend": "stable",  # Would analyze volume changes
                "price_momentum": "neutral",  # Would analyze price movements
                "liquidity_assessment": "good" if market_data.get('volume', 0) > 1000000 else "poor"
            }
        }

        # Add to results
        intelligence_data["markets_analyzed"].append(market_intelligence)

    # Generate summary
    successful_analyses = [m for m in intelligence_data["markets_analyzed"] if m["status"] == "analyzed"]

    intelligence_data["intelligence_summary"] = {
        "total_requested": len(market_ids),
        "successfully_analyzed": len(successful_analyses),
        "analysis_success_rate": len(successful_analyses) / max(1, len(market_ids)),

        "market_categories": list(set(
            m["market_data"].get("category", "unknown")
            for m in successful_analyses
            if "market_data" in m
        )),

        "volume_distribution": {
            "total_volume": sum(
                m["market_data"].get("volume", 0)
                for m in successful_analyses
                if "market_data" in m
            ),
            "avg_volume": sum(
                m["market_data"].get("volume", 0)
                for m in successful_analyses
                if "market_data" in m
            ) / max(1, len(successful_analyses))
        }
    }

    return intelligence_data


def collect_category_insights(categories: List[str], time_period: str = "30d") -> Dict[str, Any]:
    """
    Collect insights across entire market categories.

    Args:
        categories: List of categories to analyze
        time_period: Time period for analysis
    """
    analyzer = MarketAnalyzer()

    category_insights = {
        "categories_analyzed": categories,
        "time_period": time_period,
        "category_summaries": {},
        "cross_category_insights": {},
        "timestamp": datetime.now().isoformat()
    }

    for category in categories:
        # Get markets in this category
        markets = analyzer.get_high_volume_markets(category=category, limit=100)

        if not markets:
            category_insights["category_summaries"][category] = {
                "status": "no_data",
                "market_count": 0,
                "total_volume": 0
            }
            continue

        # Calculate category metrics
        total_volume = sum(m.get('volume', 0) for m in markets)
        avg_probability = sum(m.get('implied_probability', 0) for m in markets) / max(1, len(markets))

        category_summary = {
            "status": "analyzed",
            "market_count": len(markets),
            "total_volume": total_volume,
            "avg_volume_per_market": total_volume / max(1, len(markets)),
            "avg_implied_probability": avg_probability,

            "volume_distribution": {
                "largest_market_volume": max((m.get('volume', 0) for m in markets), default=0),
                "smallest_market_volume": min((m.get('volume', 0) for m in markets), default=0),
                "volume_std_dev": 0  # Would calculate standard deviation
            },

            "probability_distribution": {
                "high_confidence_markets": sum(1 for m in markets if m.get('implied_probability', 0) > 0.8 or m.get('implied_probability', 0) < 0.2),
                "moderate_confidence_markets": sum(1 for m in markets if 0.3 <= m.get('implied_probability', 0) <= 0.7)
            },

            "top_markets": sorted(markets, key=lambda x: x.get('volume', 0), reverse=True)[:5]
        }

        category_insights["category_summaries"][category] = category_summary

    # Cross-category insights
    all_summaries = category_insights["category_summaries"]
    successful_categories = [cat for cat, summary in all_summaries.items() if summary["status"] == "analyzed"]

    if len(successful_categories) > 1:
        total_volume_all = sum(all_summaries[cat]["total_volume"] for cat in successful_categories)
        avg_markets_per_category = sum(all_summaries[cat]["market_count"] for cat in successful_categories) / len(successful_categories)

        category_insights["cross_category_insights"] = {
            "total_volume_across_categories": total_volume_all,
            "categories_with_data": len(successful_categories),
            "avg_markets_per_category": avg_markets_per_category,

            "volume_leaders": sorted(
                [(cat, all_summaries[cat]["total_volume"]) for cat in successful_categories],
                key=lambda x: x[1],
                reverse=True
            )[:3],

            "market_count_leaders": sorted(
                [(cat, all_summaries[cat]["market_count"]) for cat in successful_categories],
                key=lambda x: x[1],
                reverse=True
            )[:3]
        }

    return category_insights


def create_data_collection_subagent():
    """
    Create the data collection subagent configuration.

    This subagent specializes in:
    - Market data collection and filtering
    - Intelligence gathering on specific markets
    - Category-wide data analysis
    - Data quality assessment
    """

    return {
        "name": "data-collection",
        "description": "Specializes in collecting and organizing market data from various sources. Use for gathering comprehensive market intelligence, filtering data by criteria, and preparing datasets for analysis.",

        "system_prompt": """You are a specialized data collection expert for prediction markets.

Your expertise includes:
- Market data gathering and filtering
- Intelligence collection on specific markets
- Category-wide data analysis
- Data quality assessment and organization

DATA COLLECTION PROCESS:
1. Search and filter markets based on query criteria
2. Gather detailed intelligence on specific markets
3. Analyze patterns across market categories
4. Assess data quality and completeness

OUTPUT FORMAT:
Return your response in this exact structure:

DATA COLLECTION SUMMARY
- Query: [search criteria used]
- Markets Found: [X markets collected]
- Categories: [categories represented]
- Volume Range: [min to max volume]

DATA QUALITY ASSESSMENT
- Complete Records: [X% of data is complete]
- Price Data Available: [X% have pricing info]
- Recent Data: [X% updated recently]

KEY INSIGHTS
- [Most active categories by volume]
- [Volume distribution patterns]
- [Data completeness assessment]

COLLECTION RECOMMENDATIONS
- [Additional data sources to explore]
- [Filtering criteria for quality data]
- [Categories with highest data completeness]

   Keep response under 500 words. Focus on data insights and collection strategy.""",
        "tools": [
            collect_market_data,
            gather_market_intelligence,
            collect_category_insights
        ],

        # Use a model good at data analysis and organization
        "model": "gpt-4o",  # Could be specialized for data work
    }

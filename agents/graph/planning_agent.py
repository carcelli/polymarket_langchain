"""
Planning Agent for Statistical Betting on Polymarket

This agent orchestrates the full betting workflow:
1. Research: Gather information and context
2. Stats: Calculate probabilities, edge, and position sizing
3. Decision: Make informed betting recommendations

Architecture:
    ┌─────────────────┐
    │   User Query    │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Research Node  │  ← Gather news, data, sentiment
    │  "What do I     │
    │   need to know?"│
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │   Stats Node    │  ← Calculate edge, EV, Kelly
    │  "What are the  │
    │   numbers?"     │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │ Probability Node│  ← LLM estimates true prob
    │  "What's the    │
    │   real chance?" │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Decision Node  │  ← Final recommendation
    │  "Should I bet?"│
    └─────────────────┘

Usage:
    from agents.graph.planning_agent import analyze_bet, find_value_opportunities

    # Analyze a specific market
    result = analyze_bet("Will Trump win 2028?")

    # Find value bets
    opportunities = find_value_opportunities(category="politics")
"""

import os
import json
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langsmith import traceable

from agents.memory.manager import MemoryManager


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class BetRecommendation:
    """A structured betting recommendation."""

    market_id: str
    market_question: str
    side: str  # YES or NO
    confidence: float  # 0-1
    estimated_prob: float  # Our estimate
    market_prob: float  # Current market price
    edge: float  # estimated - market
    expected_value: float
    kelly_fraction: float
    suggested_size: float  # Dollar amount
    reasoning: str
    risk_factors: List[str]


# =============================================================================
# STATE DEFINITION
# =============================================================================


class PlanningState(TypedDict):
    """State for the planning agent."""

    messages: Annotated[List[BaseMessage], add_messages]

    # Input
    query: str
    target_market_id: Optional[str]

    # Research phase outputs
    market_data: Dict[str, Any]
    research_context: Dict[str, Any]
    news_sentiment: Dict[str, Any]

    # Stats phase outputs
    implied_probability: float
    price_history: List[Dict]
    volume_analysis: Dict[str, Any]

    # Probability estimation
    estimated_probability: float
    probability_reasoning: str

    # Final decision
    edge: float
    expected_value: float
    kelly_fraction: float
    recommendation: Dict[str, Any]

    # Error handling
    error: Optional[str]


# =============================================================================
# NODE IMPLEMENTATIONS
# =============================================================================


@traceable(name="research_node", run_type="retriever")
def research_node(state: PlanningState) -> Dict:
    """
    Research Node: Gather all relevant information.

    - Fetches market data from local DB
    - Gets any stored research
    - Analyzes volume and liquidity
    """
    query = state.get("query", "")
    target_id = state.get("target_market_id")

    print("📚 Research Node: Gathering market intelligence...")

    try:
        mm = MemoryManager("data/markets.db")

        research_context = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
        }

        # Find the target market
        market_data = None

        if target_id:
            market_data = mm.get_market(target_id)

        if not market_data:
            # Search for market - try full query first, then individual words
            results = mm.search_markets(query, limit=5)

            if not results:
                # Try searching with individual significant words
                words = [w for w in query.split() if len(w) > 3]
                for word in words:
                    results = mm.search_markets(word, limit=5)
                    if results:
                        break

            if results:
                market_data = results[0]
                print(f"   📍 Found market: {market_data['question'][:50]}...")

        if not market_data:
            return {
                "error": f"Could not find market for: {query}",
                "research_context": research_context,
            }

        # Get stored research
        stored_research = mm.get_market_research(market_data["id"])
        research_context["stored_research"] = stored_research

        # Get related markets (same category)
        category = market_data.get("category", "other")
        related = mm.list_markets_by_category(category, limit=5)
        research_context["related_markets"] = [
            {"question": m["question"][:60], "volume": m.get("volume", 0)}
            for m in related
            if m["id"] != market_data["id"]
        ]

        # Volume analysis
        total_category_vol = sum(m.get("volume", 0) for m in related)
        research_context["category_volume"] = total_category_vol
        research_context["market_share"] = (
            market_data.get("volume", 0) / total_category_vol * 100
            if total_category_vol > 0
            else 0
        )

        # Get database stats for context
        stats = mm.get_stats()
        research_context["database_stats"] = stats

        print(f"   ✅ Market: {market_data['question'][:40]}...")
        print(f"   📊 Volume: ${market_data.get('volume', 0):,.0f}")
        print(f"   📁 Category: {category} ({len(related)} related)")

        return {
            "market_data": market_data,
            "research_context": research_context,
            "target_market_id": market_data["id"],
            "messages": [
                AIMessage(
                    content=f"Research complete for: {market_data['question'][:50]}..."
                )
            ],
        }

    except Exception as e:
        print(f"   ❌ Research error: {str(e)}")
        return {"error": f"Research failed: {str(e)}"}


@traceable(name="stats_node", run_type="tool")
def stats_node(state: PlanningState) -> Dict:
    """
    Stats Node: Calculate betting statistics.

    - Extract implied probability from market price
    - Analyze price history/momentum
    - Calculate volume metrics
    """
    market_data = state.get("market_data", {})

    # Check for upstream errors
    if state.get("error"):
        print(f"   ⏭️  Skipping stats: {state.get('error')}")
        return {}

    if not market_data:
        return {"error": "No market data available for stats"}

    print("📊 Stats Node: Crunching numbers...")

    try:
        mm = MemoryManager("data/markets.db")
        market_id = market_data.get("id")

        # Get prices
        prices = market_data.get("outcome_prices", [])
        implied_prob = 0.5

        if prices and len(prices) >= 2:
            try:
                implied_prob = float(prices[0])
            except (ValueError, TypeError):
                pass

        # Get price history
        price_history = mm.get_price_history(market_id, hours=72)
        price_change = mm.get_price_change(market_id, hours=24)

        # Volume analysis
        volume = market_data.get("volume", 0)
        liquidity = market_data.get("liquidity", 0)

        volume_analysis = {
            "total_volume": volume,
            "liquidity": liquidity,
            "volume_to_liquidity": volume / liquidity if liquidity > 0 else 0,
            "price_momentum": price_change.get("change", 0) if price_change else 0,
            "price_momentum_pct": (
                price_change.get("change_pct", 0) if price_change else 0
            ),
        }

        # Determine confidence in market price (higher volume = more reliable)
        if volume > 1_000_000:
            volume_analysis["price_confidence"] = "HIGH"
        elif volume > 100_000:
            volume_analysis["price_confidence"] = "MEDIUM"
        else:
            volume_analysis["price_confidence"] = "LOW"

        print(f"   📈 Implied Prob: {implied_prob:.1%}")
        print(f"   💰 Volume: ${volume:,.0f}")
        print(f"   📉 24h Change: {volume_analysis['price_momentum_pct']:.1f}%")

        return {
            "implied_probability": implied_prob,
            "price_history": price_history,
            "volume_analysis": volume_analysis,
            "messages": [
                AIMessage(
                    content=f"Stats: Implied prob {implied_prob:.1%}, volume ${volume:,.0f}"
                )
            ],
        }

    except Exception as e:
        print(f"   ❌ Stats error: {str(e)}")
        return {"error": f"Stats calculation failed: {str(e)}"}


@traceable(name="probability_estimation", run_type="llm")
def probability_node(state: PlanningState) -> Dict:
    """
    Probability Node: LLM + ML estimates true probability.

    Uses research context, stats, and optionally XGBoost model to estimate
    the "real" probability, which is then compared to market price to find edge.
    """
    # Check for upstream errors
    if state.get("error"):
        print(f"   ⏭️  Skipping probability: {state.get('error')}")
        return {"error": state.get("error")}

    print("🎯 Probability Node: Estimating true probability...")

    market_data = state.get("market_data", {})
    research_context = state.get("research_context", {})
    volume_analysis = state.get("volume_analysis", {})
    implied_prob = state.get("implied_probability", 0.5)

    if not market_data:
        return {"error": "No market data for probability estimation"}

    try:
        # Try to get XGBoost prediction first
        xgboost_prob = None
        xgboost_reasoning = None

        try:
            from agents.ml_strategies.xgboost_strategy import XGBoostProbabilityStrategy

            # Check if model exists and is trained
            model_path = "data/models/xgboost_probability_model.json"
            if os.path.exists(model_path):
                xgboost_strategy = XGBoostProbabilityStrategy()
                xgboost_strategy.load_model(model_path)

                # Make prediction
                xgboost_result = xgboost_strategy.predict(market_data)
                xgboost_prob = xgboost_result.predicted_probability
                xgboost_reasoning = f"XGBoost predicts {xgboost_prob:.3f} (edge: {xgboost_result.edge:.1%})"
                print(f"   🤖 XGBoost Prob: {xgboost_prob:.1%}")

        except Exception as e:
            print(f"   ⚠️  XGBoost unavailable: {e}")
            xgboost_prob = None

        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,  # Lower temp for more consistent estimates
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        # Build context
        question = market_data.get("question", "Unknown")
        description = market_data.get("description", "")[:500]
        category = market_data.get("category", "other")
        end_date = market_data.get("end_date", "Unknown")

        stored_research = research_context.get("stored_research", [])
        research_summary = ""
        if stored_research:
            research_summary = "\n".join(
                [
                    f"- [{r.get('research_type')}] {r.get('content', '')[:200]}"
                    for r in stored_research[:3]
                ]
            )

        system_prompt = """You are an expert probability estimator for prediction markets.

Your task is to estimate the TRUE probability of an event, which may differ from the market price.

Guidelines:
1. Consider base rates and historical data
2. Account for recent news and developments
3. Factor in time until resolution
4. Be calibrated - avoid overconfidence
5. Express genuine uncertainty
6. If ML model prediction is provided, use it as a strong signal but apply your judgment

Output format:
PROBABILITY: [0.XX]
CONFIDENCE: [LOW/MEDIUM/HIGH]
REASONING: [2-3 sentences explaining your estimate]

Be specific and data-driven. If highly uncertain, lean toward 0.50."""

        # Include XGBoost prediction in the prompt if available
        ml_context = ""
        if xgboost_prob is not None:
            ml_context = f"\nMACHINE LEARNING PREDICTION: XGBoost model estimates {xgboost_prob:.1%} probability of YES."

        user_prompt = f"""Estimate the probability for this prediction market:

QUESTION: {question}

DESCRIPTION: {description}

CATEGORY: {category}
END DATE: {end_date}
CURRENT MARKET PRICE: {implied_prob:.1%} (YES)
VOLUME: ${market_data.get('volume', 0):,.0f}
PRICE CONFIDENCE: {volume_analysis.get('price_confidence', 'UNKNOWN')}

{f'RESEARCH NOTES:{chr(10)}{research_summary}' if research_summary else ''}{ml_context}

Based on your knowledge and the above context, what is the TRUE probability of YES?"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        response_text = response.content

        # Parse probability from response
        estimated_prob = implied_prob  # Default to market if parsing fails
        reasoning = response_text

        if "PROBABILITY:" in response_text:
            try:
                prob_line = [
                    l for l in response_text.split("\n") if "PROBABILITY:" in l
                ][0]
                prob_str = prob_line.split(":")[1].strip()
                estimated_prob = float(prob_str.replace("%", "").strip())
                if estimated_prob > 1:
                    estimated_prob /= 100  # Convert from percentage
            except:
                pass

        print(f"   🎲 Estimated Prob: {estimated_prob:.1%}")
        print(f"   📊 Market Price: {implied_prob:.1%}")
        print(f"   ⚡ Raw Edge: {(estimated_prob - implied_prob)*100:.1f}%")

        # Combine reasoning with XGBoost info if available
        combined_reasoning = reasoning
        if xgboost_reasoning:
            combined_reasoning = f"{xgboost_reasoning}\n\n{reasoning}"

        return {
            "estimated_probability": estimated_prob,
            "probability_reasoning": combined_reasoning,
            "messages": [response],
            "xgboost_probability": xgboost_prob,
            "xgboost_reasoning": xgboost_reasoning,
        }

    except Exception as e:
        print(f"   ❌ Probability estimation error: {str(e)}")
        return {
            "estimated_probability": implied_prob,
            "probability_reasoning": f"Error: {str(e)}, using market price",
            "error": str(e),
        }


@traceable(name="decision_node")
def decision_node(state: PlanningState) -> Dict:
    """
    Decision Node: Make final betting recommendation.

    Combines all analysis to produce:
    - Edge calculation
    - Expected value
    - Kelly criterion position sizing
    - Final recommendation
    """
    print("💡 Decision Node: Forming recommendation...")

    market_data = state.get("market_data", {})
    implied_prob = state.get("implied_probability", 0.5)
    estimated_prob = state.get("estimated_probability", 0.5)
    reasoning = state.get("probability_reasoning", "")
    volume_analysis = state.get("volume_analysis", {})

    if not market_data:
        return {"error": "No market data for decision"}

    # Calculate edge
    edge_yes = estimated_prob - implied_prob
    edge_no = (1 - estimated_prob) - (1 - implied_prob)

    # Determine best side
    if abs(edge_yes) > abs(edge_no):
        side = "YES"
        edge = edge_yes
        our_prob = estimated_prob
        market_prob = implied_prob
    else:
        side = "NO"
        edge = edge_no
        our_prob = 1 - estimated_prob
        market_prob = 1 - implied_prob

    # Expected Value
    # EV = (prob * profit) - ((1-prob) * loss)
    # For binary: EV = prob * (1 - price) - (1 - prob) * price
    ev = our_prob * (1 - market_prob) - (1 - our_prob) * market_prob

    # Kelly Criterion
    # f* = (bp - q) / b, where b = odds, p = our prob, q = 1-p
    kelly = 0
    if market_prob > 0 and market_prob < 1:
        b = (1 - market_prob) / market_prob
        kelly = (b * our_prob - (1 - our_prob)) / b
        kelly = max(0, min(kelly, 0.25))  # Cap at 25%

    # Build recommendation
    recommendation = {
        "market_id": market_data.get("id"),
        "market_question": market_data.get("question"),
        "recommended_side": side,
        "edge": edge,
        "expected_value": ev,
        "kelly_fraction": kelly,
        "estimated_probability": estimated_prob,
        "market_probability": implied_prob,
        "volume": market_data.get("volume", 0),
        "category": market_data.get("category"),
        "reasoning": reasoning,
        "timestamp": datetime.now().isoformat(),
    }

    # Determine action
    MIN_EDGE = 0.03  # 3% minimum edge
    MIN_VOLUME = 5000  # $5k minimum volume

    if edge > MIN_EDGE and market_data.get("volume", 0) > MIN_VOLUME and kelly > 0.01:
        recommendation["action"] = "BET"
        recommendation["suggested_kelly"] = kelly
        recommendation["confidence"] = "HIGH" if edge > 0.10 else "MEDIUM"
    elif edge > 0 and kelly > 0:
        recommendation["action"] = "WATCH"
        recommendation["confidence"] = "LOW"
    else:
        recommendation["action"] = "PASS"
        recommendation["confidence"] = "N/A"

    # Risk factors
    risk_factors = []
    if market_data.get("volume", 0) < 50000:
        risk_factors.append("Low volume - price may not be reliable")
    if abs(edge) > 0.20:
        risk_factors.append("Large edge - verify reasoning carefully")
    if volume_analysis.get("price_confidence") == "LOW":
        risk_factors.append("Low liquidity market")

    recommendation["risk_factors"] = risk_factors

    # Print summary
    print(f"\n   {'='*50}")
    print(f"   📋 RECOMMENDATION: {recommendation['action']}")
    print(f"   {'='*50}")
    print(f"   Market: {market_data.get('question', '')[:50]}...")
    print(f"   Side: {side}")
    print(f"   Edge: {edge*100:.1f}%")
    print(f"   EV: {ev*100:.2f}%")
    print(f"   Kelly: {kelly*100:.1f}%")
    if risk_factors:
        print(f"   ⚠️  Risks: {', '.join(risk_factors)}")
    print(f"   {'='*50}\n")

    # Store analytics
    mm = MemoryManager("data/markets.db")
    mm.update_market_analytics(
        market_data.get("id"),
        estimated_prob=estimated_prob,
        analyst_notes=f"Action: {recommendation['action']}, Edge: {edge:.2%}",
    )

    return {
        "edge": edge,
        "expected_value": ev,
        "kelly_fraction": kelly,
        "recommendation": recommendation,
    }


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================


def create_planning_agent():
    """
    Create the planning agent graph.

    Flow: Research → Stats → Probability → Decision
    """
    print("🏗️  Building Planning Agent...")

    workflow = StateGraph(PlanningState)

    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("stats", stats_node)
    workflow.add_node("probability", probability_node)
    workflow.add_node("decision", decision_node)

    # Define flow
    workflow.set_entry_point("research")
    workflow.add_edge("research", "stats")
    workflow.add_edge("stats", "probability")
    workflow.add_edge("probability", "decision")
    workflow.add_edge("decision", END)

    graph = workflow.compile()
    print("   ✅ Planning Agent ready")

    return graph


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


@traceable(name="analyze_bet")
def analyze_bet(query: str, market_id: str = None) -> Dict:
    """
    Analyze a betting opportunity.

    Args:
        query: Market question or search term
        market_id: Optional specific market ID

    Returns:
        Full analysis with recommendation
    """
    graph = create_planning_agent()

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "target_market_id": market_id,
        "market_data": {},
        "research_context": {},
        "news_sentiment": {},
        "implied_probability": 0.5,
        "price_history": [],
        "volume_analysis": {},
        "estimated_probability": 0.5,
        "probability_reasoning": "",
        "edge": 0,
        "expected_value": 0,
        "kelly_fraction": 0,
        "recommendation": {},
        "error": None,
    }

    result = graph.invoke(initial_state)
    return result


@traceable(name="find_value_opportunities")
def find_value_opportunities(
    category: str = None, min_volume: float = 10000, limit: int = 5
) -> List[Dict]:
    """
    Scan markets for potential value betting opportunities.

    Args:
        category: Optional category filter
        min_volume: Minimum volume requirement
        limit: Number of markets to analyze

    Returns:
        List of recommendations sorted by edge
    """
    print(f"\n🔍 Scanning for value opportunities...")
    if category:
        print(f"   Category: {category}")
    print(f"   Min Volume: ${min_volume:,.0f}")

    mm = MemoryManager("data/markets.db")

    # Get candidate markets
    if category:
        candidates = mm.list_markets_by_category(category, limit=limit * 2)
    else:
        candidates = mm.list_top_volume_markets(limit=limit * 2)

    # Filter by volume
    candidates = [m for m in candidates if m.get("volume", 0) >= min_volume][:limit]

    print(f"   Found {len(candidates)} candidates\n")

    opportunities = []
    for market in candidates:
        print(f"\n{'─'*60}")
        result = analyze_bet(market["question"], market["id"])

        if result.get("recommendation", {}).get("action") in ["BET", "WATCH"]:
            opportunities.append(result["recommendation"])

    # Sort by edge
    opportunities.sort(key=lambda x: abs(x.get("edge", 0)), reverse=True)

    print(f"\n{'='*60}")
    print(f"📊 OPPORTUNITY SUMMARY: {len(opportunities)} found")
    print(f"{'='*60}")

    for i, opp in enumerate(opportunities, 1):
        action_emoji = "🟢" if opp["action"] == "BET" else "🟡"
        print(f"{i}. {action_emoji} {opp['market_question'][:45]}...")
        print(
            f"   {opp['recommended_side']} | Edge: {opp['edge']*100:.1f}% | Kelly: {opp['kelly_fraction']*100:.1f}%"
        )

    return opportunities


def quick_bet_analysis(query: str) -> str:
    """Quick one-liner for bet analysis."""
    result = analyze_bet(query)
    rec = result.get("recommendation", {})

    if not rec:
        return "Could not analyze market"

    return f"""
Market: {rec.get('market_question', 'Unknown')[:60]}...
Action: {rec.get('action', 'UNKNOWN')}
Side: {rec.get('recommended_side', 'N/A')}
Edge: {rec.get('edge', 0)*100:.1f}%
Kelly: {rec.get('kelly_fraction', 0)*100:.1f}%
"""


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--scan":
            # Scan for opportunities
            category = sys.argv[2] if len(sys.argv) > 2 else None
            find_value_opportunities(category=category)
        else:
            # Analyze specific market
            query = " ".join(sys.argv[1:])
            result = analyze_bet(query)

            if result.get("error"):
                print(f"\n❌ Error: {result['error']}")
    else:
        # Default: scan top markets
        print("Usage:")
        print("  python planning_agent.py 'Will Trump win?'  # Analyze specific")
        print("  python planning_agent.py --scan             # Scan all")
        print("  python planning_agent.py --scan politics    # Scan category")

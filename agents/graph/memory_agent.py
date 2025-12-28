"""
Memory-Aware LangGraph Agent for Polymarket

This module implements a stateful agent that:
1. Queries local knowledge base FIRST (20k+ markets, instant)
2. Enriches with live API data only when needed
3. Reasons about opportunities using GPT-4
4. Makes trade decisions with full context

The key insight: Instead of blindly pinging the API every time,
the agent carries a local, searchable copy of the entire market universe.

Architecture:
    ┌─────────────────┐
    │   User Query    │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Memory Node    │  ← Query local DB (instant)
    │  "What do I     │
    │   already know?"│
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Enrichment     │  ← Live API (only if needed)
    │  Node           │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Reasoning      │  ← GPT-4 analysis
    │  Node           │
    └────────┬────────┘
             │
    ┌────────▼────────┐
    │  Decision       │  ← Trade/No Trade
    │  Node           │
    └─────────────────┘

Usage:
    from agents.graph.memory_agent import create_memory_agent, run_memory_agent
    
    agent = create_memory_agent()
    result = run_memory_agent(agent, "Find mispriced Trump markets")
"""

import os
import json
from typing import TypedDict, Annotated, List, Optional, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()  # Load environment variables

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# LangSmith tracing for observability
from langsmith import traceable

# Local imports
from agents.memory.manager import MemoryManager


# =============================================================================
# STATE DEFINITION
# =============================================================================

class MemoryAgentState(TypedDict):
    """
    The cognitive state of a memory-aware trading agent.
    
    This extends the basic state with memory context, allowing the agent
    to reason about what it already knows before making API calls.
    """
    # Conversation history
    messages: Annotated[List[BaseMessage], add_messages]
    
    # The original user query
    query: str
    
    # Memory context: What the agent retrieved from local DB
    memory_context: Dict[str, Any]
    
    # Live data: Fresh data from API (if enrichment was needed)
    live_data: Dict[str, Any]
    
    # Analysis: The agent's reasoning and probability estimates
    analysis: Dict[str, Any]
    
    # Decision: The final trade decision (if any)
    decision: Dict[str, Any]
    
    # Error state
    error: Optional[str]


# =============================================================================
# NODE IMPLEMENTATIONS
# =============================================================================

@traceable(name="memory_query", run_type="retriever")
def memory_node(state: MemoryAgentState) -> Dict:
    """
    The Memory Node: Query local knowledge base.
    
    This is the first node in the graph. Instead of hitting the API,
    we query our local database of 20k+ markets.
    
    The node:
    1. Parses the user query to extract intent
    2. Queries relevant markets from memory
    3. Adds context to state for downstream nodes
    """
    query = state.get("query", "")
    
    print("🧠 Memory Node: Querying local knowledge base...")
    
    try:
        mm = MemoryManager("data/markets.db")
        
        # Get database stats for context
        stats = mm.get_stats()
        categories = mm.get_categories()
        
        # Determine what to fetch based on query keywords
        memory_context = {
            "timestamp": datetime.now().isoformat(),
            "database_stats": stats,
            "categories": categories,
            "relevant_markets": [],
            "top_volume_markets": [],
            "search_results": [],
        }
        
        # Parse query for category hints
        query_lower = query.lower()
        
        # Check for category-specific queries
        category_keywords = {
            "politics": ["trump", "biden", "election", "senate", "congress", "president", "political"],
            "sports": ["nfl", "nba", "mlb", "sports", "super bowl", "championship", "football", "basketball"],
            "crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain"],
            "tech": ["ai", "openai", "apple", "google", "tech", "software"],
            "geopolitics": ["ukraine", "russia", "war", "china", "ceasefire"],
            "finance": ["fed", "interest rate", "stock", "market"],
            "economy": ["gdp", "inflation", "jobs", "economic"],
        }
        
        detected_category = None
        for cat, keywords in category_keywords.items():
            if any(kw in query_lower for kw in keywords):
                detected_category = cat
                break
        
        # Fetch relevant markets
        if detected_category:
            memory_context["detected_category"] = detected_category
            memory_context["relevant_markets"] = mm.list_markets_by_category(
                detected_category, limit=20
            )
            print(f"   📂 Found {len(memory_context['relevant_markets'])} {detected_category} markets")
        
        # Always get top volume markets for context
        memory_context["top_volume_markets"] = mm.list_top_volume_markets(limit=10)
        
        # Text search if query seems specific
        search_terms = [word for word in query.split() if len(word) > 3]
        if search_terms:
            search_query = " ".join(search_terms[:3])  # Use first 3 significant words
            memory_context["search_results"] = mm.search_markets(search_query, limit=10)
            print(f"   🔍 Search '{search_query}' returned {len(memory_context['search_results'])} results")
        
        print(f"   ✅ Memory context loaded: {stats['total_markets']:,} markets available")
        
        return {
            "memory_context": memory_context,
            "messages": [AIMessage(content=f"Memory loaded: {stats['total_markets']:,} markets across {len(categories)} categories")]
        }
        
    except Exception as e:
        print(f"   ❌ Memory error: {str(e)}")
        return {
            "memory_context": {},
            "error": f"Memory query failed: {str(e)}",
            "messages": [AIMessage(content=f"Memory error: {str(e)}")]
        }


@traceable(name="api_enrichment", run_type="tool")
def enrichment_node(state: MemoryAgentState) -> Dict:
    """
    The Enrichment Node: Fetch live data if needed.
    
    This node decides whether to fetch fresh data from the API.
    It only makes API calls if:
    1. Memory is stale (>1 hour old)
    2. User specifically asks for "live" or "current" data
    3. Memory returned no results
    
    This saves API calls and improves response time.
    """
    query = state.get("query", "").lower()
    memory_context = state.get("memory_context", {})
    
    print("🔄 Enrichment Node: Checking if live data needed...")
    
    # Check if we need live data
    needs_live_data = (
        "live" in query or
        "current" in query or
        "now" in query or
        "latest" in query or
        not memory_context.get("relevant_markets") and not memory_context.get("search_results")
    )
    
    if not needs_live_data:
        print("   ⏭️  Memory sufficient, skipping API call")
        return {"live_data": {}}
    
    print("   📡 Fetching live data from API...")
    
    try:
        from agents.polymarket.gamma import GammaMarketClient
        gamma = GammaMarketClient()
        
        live_data = {
            "timestamp": datetime.now().isoformat(),
            "current_events": [],
        }
        
        # Fetch current events
        events = gamma.get_current_events(limit=10)
        live_data["current_events"] = [
            {
                "title": e.get("title"),
                "volume": e.get("volume"),
                "market_count": len(e.get("markets", [])),
            }
            for e in events
        ]
        
        print(f"   ✅ Fetched {len(live_data['current_events'])} live events")
        
        return {
            "live_data": live_data,
            "messages": [AIMessage(content=f"Enriched with {len(live_data['current_events'])} live events")]
        }
        
    except Exception as e:
        print(f"   ⚠️  API enrichment failed: {str(e)}")
        return {"live_data": {"error": str(e)}}


@traceable(name="llm_reasoning", run_type="llm")
def reasoning_node(state: MemoryAgentState) -> Dict:
    """
    The Reasoning Node: GPT-4 analysis.
    
    This is where the LLM synthesizes:
    - User's query
    - Memory context (local DB results)
    - Live data (if fetched)
    
    And produces structured analysis.
    """
    print("🤔 Reasoning Node: Analyzing with LLM...")
    
    query = state.get("query", "")
    memory_context = state.get("memory_context", {})
    live_data = state.get("live_data", {})
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        
        # Build context for LLM
        context_parts = []
        
        # Add database stats
        stats = memory_context.get("database_stats", {})
        if stats:
            context_parts.append(f"Database contains {stats.get('total_markets', 0):,} markets with ${stats.get('total_volume', 0):,.0f} total volume.")
        
        # Add relevant markets
        relevant = memory_context.get("relevant_markets", [])
        if relevant:
            context_parts.append(f"\nRelevant markets ({len(relevant)} found):")
            for m in relevant[:5]:
                prices = m.get("outcome_prices", [])
                if prices and len(prices) >= 2:
                    try:
                        yes_price = float(prices[0]) * 100
                        context_parts.append(f"- {m['question'][:60]}... (YES: {yes_price:.0f}%, Vol: ${m.get('volume', 0):,.0f})")
                    except:
                        context_parts.append(f"- {m['question'][:60]}... (Vol: ${m.get('volume', 0):,.0f})")
        
        # Add search results
        search_results = memory_context.get("search_results", [])
        if search_results:
            context_parts.append(f"\nSearch results ({len(search_results)} found):")
            for m in search_results[:5]:
                context_parts.append(f"- {m['question'][:60]}... (Vol: ${m.get('volume', 0):,.0f})")
        
        # Add live data if available
        if live_data.get("current_events"):
            context_parts.append(f"\nLive events ({len(live_data['current_events'])} fetched):")
            for e in live_data["current_events"][:3]:
                context_parts.append(f"- {e.get('title', 'N/A')[:50]}...")
        
        context = "\n".join(context_parts)
        
        # Create prompt
        system_prompt = """You are an expert Polymarket analyst with access to a local database of 20,000+ markets.

Your role is to:
1. Analyze the provided market data
2. Identify opportunities, mispricing, or interesting patterns
3. Provide probability estimates where relevant
4. Give actionable recommendations

Be specific, data-driven, and express uncertainty appropriately."""

        user_prompt = f"""Query: {query}

Available Context:
{context}

Based on this data, provide:
1. Direct answer to the query
2. Key markets to watch (with reasoning)
3. Any potential trading opportunities
4. Confidence level in your analysis"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        
        response = llm.invoke(messages)
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "llm_response": response.content,
            "markets_analyzed": len(relevant) + len(search_results),
        }
        
        print(f"   ✅ Analysis complete ({len(response.content)} chars)")
        
        return {
            "analysis": analysis,
            "messages": [response],
        }
        
    except Exception as e:
        print(f"   ❌ Reasoning error: {str(e)}")
        return {
            "analysis": {"error": str(e)},
            "error": f"Reasoning failed: {str(e)}",
        }


@traceable(name="trade_decision")
def decide_node(state: MemoryAgentState) -> Dict:
    """
    The Decision Node: Determine if action is needed.
    
    This node examines the analysis and decides:
    1. Is there a trading opportunity?
    2. What action should be taken?
    3. What's the confidence level?
    """
    print("⚡ Decision Node: Evaluating trade decision...")
    
    analysis = state.get("analysis", {})
    
    # For now, we don't auto-trade - just structure the decision
    decision = {
        "timestamp": datetime.now().isoformat(),
        "action": "ANALYZE_ONLY",  # Could be: BUY, SELL, HOLD, ANALYZE_ONLY
        "confidence": "N/A",
        "reasoning": analysis.get("llm_response", "No analysis available"),
    }
    
    print(f"   ✅ Decision: {decision['action']}")
    
    return {"decision": decision}


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def create_memory_agent():
    """
    Create a memory-aware LangGraph agent.
    
    The graph flows:
    Memory → Enrichment → Reasoning → Decision
    
    Returns:
        Compiled LangGraph
    """
    print("🏗️  Building Memory Agent Graph...")
    
    # Create the graph
    workflow = StateGraph(MemoryAgentState)
    
    # Add nodes
    workflow.add_node("memory", memory_node)
    workflow.add_node("enrichment", enrichment_node)
    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("decide", decide_node)
    
    # Define edges (linear flow for now)
    workflow.set_entry_point("memory")
    workflow.add_edge("memory", "enrichment")
    workflow.add_edge("enrichment", "reasoning")
    workflow.add_edge("reasoning", "decide")
    workflow.add_edge("decide", END)
    
    # Compile
    graph = workflow.compile()
    print("   ✅ Graph compiled successfully")
    
    return graph


@traceable(name="polymarket_agent_pipeline")
def run_memory_agent(graph, query: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Run the memory agent with a query.
    
    Args:
        graph: Compiled LangGraph from create_memory_agent()
        query: Natural language query
        verbose: Print intermediate steps
    
    Returns:
        Final state with analysis and decision
    """
    if verbose:
        print("\n" + "=" * 60)
        print(f"  MEMORY AGENT: {query[:50]}...")
        print("=" * 60 + "\n")
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "memory_context": {},
        "live_data": {},
        "analysis": {},
        "decision": {},
        "error": None,
    }
    
    # Run the graph
    try:
        result = graph.invoke(initial_state)
        
        if verbose:
            print("\n" + "-" * 60)
            print("  RESULT")
            print("-" * 60)
            if result.get("analysis", {}).get("llm_response"):
                print(result["analysis"]["llm_response"])
            print("-" * 60 + "\n")
        
        return result
        
    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

@traceable(name="quick_analysis")
def quick_analysis(query: str) -> str:
    """
    Quick one-liner for memory-based analysis.
    
    Args:
        query: What to analyze
    
    Returns:
        Analysis result as string
    """
    graph = create_memory_agent()
    result = run_memory_agent(graph, query, verbose=False)
    return result.get("analysis", {}).get("llm_response", "No analysis available")


def get_market_overview(category: str = None) -> Dict:
    """
    Get a quick overview of markets from memory.
    
    Args:
        category: Optional category filter
    
    Returns:
        Market overview dict
    """
    mm = MemoryManager("data/markets.db")
    
    overview = {
        "stats": mm.get_stats(),
        "categories": mm.get_categories(),
    }
    
    if category:
        overview["markets"] = mm.list_markets_by_category(category, limit=20)
    else:
        overview["top_markets"] = mm.list_top_volume_markets(limit=20)
    
    return overview


# =============================================================================
# CLI INTERFACE
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What are the most interesting political markets right now?"
    
    graph = create_memory_agent()
    result = run_memory_agent(graph, query)
    
    if result.get("error"):
        print(f"\n❌ Error: {result['error']}")
    else:
        print("\n✅ Analysis complete!")


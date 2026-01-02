import sys
import os
from pathlib import Path

# Add the project root to Python path BEFORE any other imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import typer
from devtools import pprint

app = typer.Typer()


@app.command()
def get_all_markets(limit: int = 5, sort_by: str = "spread") -> None:
    """
    Query Polymarket's markets
    """
    print(f"limit: int = {limit}, sort_by: str = {sort_by}")
    from agents.polymarket.polymarket import Polymarket

    polymarket = Polymarket()
    markets = polymarket.get_all_markets()
    markets = polymarket.filter_markets_for_trading(markets)
    if sort_by == "spread":
        markets = sorted(markets, key=lambda x: x.spread, reverse=True)
    markets = markets[:limit]
    pprint(markets)


@app.command()
def get_relevant_news(keywords: str) -> None:
    """
    Use NewsAPI to query the internet
    """
    from agents.connectors.news import News

    newsapi_client = News()
    articles = newsapi_client.get_articles_for_cli_keywords(keywords)
    pprint(articles)


@app.command()
def get_all_events(
    limit: int = 5, sort_by: str = "number_of_markets", include_closed: bool = False
) -> None:
    """
    Query Polymarket's events
    """
    print(
        f"limit: int = {limit}, sort_by: str = {sort_by}, include_closed: {include_closed}"
    )
    from agents.polymarket.polymarket import Polymarket

    polymarket = Polymarket()
    events = polymarket.get_all_events()
    if not include_closed:
        events = polymarket.filter_events_for_trading(events)
    if sort_by == "number_of_markets":
        events = sorted(events, key=lambda x: len(x.markets.split(",")), reverse=True)
    events = events[:limit]
    pprint(events)


@app.command()
def create_local_markets_rag(local_directory: str) -> None:
    """
    Create a local markets database for RAG
    """
    from agents.connectors.chroma import PolymarketRAG

    polymarket_rag = PolymarketRAG()
    polymarket_rag.create_local_markets_rag(local_directory=local_directory)


@app.command()
def query_local_markets_rag(vector_db_directory: str, query: str) -> None:
    """
    RAG over a local database of Polymarket's events
    """
    from agents.connectors.chroma import PolymarketRAG

    polymarket_rag = PolymarketRAG()
    response = polymarket_rag.query_local_markets_rag(
        local_directory=vector_db_directory, query=query
    )
    pprint(response)


@app.command()
def ask_superforecaster(event_title: str, market_question: str, outcome: str) -> None:
    """
    Ask a superforecaster about a trade
    """
    print(
        f"event: str = {event_title}, question: str = {market_question}, outcome (usually yes or no): str = {outcome}"
    )
    from agents.application.executor import Executor

    executor = Executor()
    response = executor.get_superforecast(
        event_title=event_title, market_question=market_question, outcome=outcome
    )
    print(f"Response:{response}")


@app.command()
def create_market() -> None:
    """
    Format a request to create a market on Polymarket
    """
    from agents.application.creator import Creator

    c = Creator()
    market_description = c.one_best_market()
    print(f"market_description: str = {market_description}")


@app.command()
def ask_llm(user_input: str) -> None:
    """
    Ask a question to the LLM and get a response.
    """
    from agents.application.executor import Executor

    executor = Executor()
    response = executor.get_llm_response(user_input)
    print(f"LLM Response: {response}")


@app.command()
def ask_polymarket_llm(user_input: str) -> None:
    """
    What types of markets do you want trade?
    """
    from agents.application.executor import Executor

    executor = Executor()
    response = executor.get_polymarket_llm(user_input=user_input)
    print(f"LLM + current markets&events response: {response}")


@app.command()
def run_autonomous_trader() -> None:
    """
    Let an autonomous system trade for you.
    """
    from agents.application.trade import Trader

    trader = Trader()
    trader.one_best_trade()


# =============================================================================
# ORCHESTRATOR AGENT CLI
# =============================================================================


@app.command()
def run_memory_agent(
    query: str = typer.Argument(
        None, help="What to analyze (e.g., 'Find interesting political markets')"
    )
) -> None:
    """
    Run the Memory Agent - queries local database first, then enriches with live data.

    Args:
        query: What to analyze (e.g., "Find interesting political markets")
    """
    if not query:
        query = "What are the most interesting political markets right now?"

    print(f"🤖 Memory Agent analyzing: {query}")
    print("=" * 60)

    try:
        from agents.graph.memory_agent import create_memory_agent, run_memory_agent

        graph = create_memory_agent()
        result = run_memory_agent(graph, query)

        if result.get("error"):
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ Analysis complete!")

    except Exception as e:
        print(f"❌ Failed to run memory agent: {e}")


@app.command()
def run_planning_agent(
    query: str = typer.Argument(None, help="Market question to analyze"),
    market_id: str = typer.Option(None, help="Optional specific market ID to analyze"),
) -> None:
    """
    Run the Planning Agent - analyzes specific markets with statistical rigor.

    Args:
        query: Market question to analyze
        market_id: Optional specific market ID to analyze
    """
    if not query:
        query = "Will the Federal Reserve cut rates in Q1 2025?"

    print(f"📊 Planning Agent analyzing: {query}")
    if market_id:
        print(f"🎯 Market ID: {market_id}")
    print("=" * 60)

    try:
        from agents.graph.planning_agent import analyze_bet

        result = analyze_bet(query, market_id)

        if result.get("error"):
            print(f"❌ Error: {result['error']}")
        else:
            print("✅ Analysis complete!")

    except Exception as e:
        print(f"❌ Failed to run planning agent: {e}")


@app.command()
def scan_opportunities(
    category: str = typer.Option(
        None, help="Optional category filter (politics, sports, crypto, etc.)"
    )
) -> None:
    """
    Scan for trading opportunities across markets.

    Args:
        category: Optional category filter (politics, sports, crypto, etc.)
    """
    print(f"🔍 Scanning for opportunities{f' in {category}' if category else ''}")
    print("=" * 60)

    try:
        from agents.graph.planning_agent import find_value_opportunities

        opportunities = find_value_opportunities(category=category)
        print(f"📊 Found {len(opportunities)} potential opportunities")

    except Exception as e:
        print(f"❌ Failed to scan opportunities: {e}")


@app.command()
def run_deep_research_agent(
    query: str = typer.Argument(..., help="Research question or trading objective"),
    agent_type: str = typer.Option(
        "trading", help="Type of agent (trading, conservative, autonomous, research)"
    ),
) -> None:
    """
    Run the Deep Research Agent with different configurations.

    Args:
        query: Research question or trading objective
        agent_type: Type of agent (trading, conservative, autonomous, research)
    """
    agent_configs = {
        "trading": "create_trading_agent_with_approval",
        "conservative": "create_conservative_trading_agent",
        "autonomous": "create_autonomous_research_agent",
        "research": "create_polymarket_research_agent",
    }

    if agent_type not in agent_configs:
        print(f"❌ Unknown agent type: {agent_type}")
        print(f"Available types: {', '.join(agent_configs.keys())}")
        return

    print(f"🧠 Deep Research Agent ({agent_type}) analyzing: {query}")
    print("=" * 60)

    try:
        from agents.deep_research_agent import (
            create_trading_agent_with_approval,
            create_conservative_trading_agent,
            create_autonomous_research_agent,
            create_polymarket_research_agent,
        )

        # Map agent type to function
        agent_functions = {
            "trading": create_trading_agent_with_approval,
            "conservative": create_conservative_trading_agent,
            "autonomous": create_autonomous_research_agent,
            "research": create_polymarket_research_agent,
        }

        create_agent = agent_functions[agent_type]
        agent = create_agent()

        # Run the agent with the query
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})

        print("✅ Deep research complete!")

    except ImportError as e:
        print(f"❌ Missing dependencies for deep research agent: {e}")
        print("Make sure you have ANTHROPIC_API_KEY and TAVILY_API_KEY set")
    except Exception as e:
        print(f"❌ Failed to run deep research agent: {e}")


@app.command()
def list_agents() -> None:
    """
    List all available agents and their capabilities.
    """
    print("🤖 Available Orchestrator Agents")
    print("=" * 50)

    agents = {
        "Memory Agent": {
            "description": "Local-first market analysis with API enrichment",
            "command": "run-memory-agent",
            "capabilities": ["20k+ local markets", "Smart API calls", "Fast responses"],
        },
        "Planning Agent": {
            "description": "Statistical market analysis with edge calculations",
            "command": "run-planning-agent",
            "capabilities": ["Kelly criterion", "Expected value", "Risk assessment"],
        },
        "Deep Research Agent": {
            "description": "Multi-agent research with web search and analysis",
            "command": "run-deep-research-agent",
            "capabilities": [
                "Web research",
                "Sub-agent coordination",
                "Comprehensive analysis",
            ],
        },
        "Opportunity Scanner": {
            "description": "Automated opportunity discovery across markets",
            "command": "scan-opportunities",
            "capabilities": [
                "Category filtering",
                "Edge detection",
                "Portfolio scanning",
            ],
        },
    }

    for name, info in agents.items():
        print(f"\n🔹 {name}")
        print(f"   {info['description']}")
        print(f"   Command: python cli.py {info['command']}")
        print(f"   Capabilities: {', '.join(info['capabilities'])}")

    print(f"\n💡 Usage Examples:")
    print(f"   python cli.py run-memory-agent 'Find political markets'")
    print(f"   python cli.py run-planning-agent 'Will BTC hit 100k?'")
    print(f"   python cli.py scan-opportunities --category politics")


if __name__ == "__main__":
    app()

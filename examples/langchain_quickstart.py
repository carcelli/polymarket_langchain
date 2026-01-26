"""
LangChain Quickstart - Learn by Running Your Own System

This demonstrates the LangChain agents you've already built.
Run each example to see how tools, agents, and ML integrate.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Verify API key
if not os.getenv("OPENAI_API_KEY"):
    print("⚠️  Set OPENAI_API_KEY in .env file first!")
    exit(1)


def example_1_simple_query():
    """Example 1: Basic agent with tool calling."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Market Query")
    print("=" * 60)
    
    from polymarket_agents.langchain.agent import create_simple_analyst
    from langchain_core.messages import HumanMessage
    
    # Create agent (uses your 50+ tools automatically)
    agent = create_simple_analyst(model="gpt-4o-mini")
    
    # Ask a question
    query = "Find the top 3 crypto markets by volume and tell me their current prices"
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    
    print("\n🤖 Agent Response:")
    # Extract final answer from messages
    if "messages" in result:
        print(result["messages"][-1].content)
    else:
        # Fallback for legacy API
        print(result.get("output", str(result)))


def example_2_ml_prediction():
    """Example 2: Agent using ML models."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: ML-Powered Prediction")
    print("=" * 60)
    
    from polymarket_agents.langchain.agent import create_ml_forecast_comparison_agent
    from langchain_core.messages import HumanMessage
    
    # This agent compares multiple ML models
    agent = create_ml_forecast_comparison_agent()
    
    query = """
    For the highest volume BTC market:
    1. Get current market price
    2. Get ML prediction from XGBoost
    3. Calculate the edge (ML prob - market prob)
    4. Recommend BUY YES, BUY NO, or PASS
    """
    
    print(f"\n📊 Query: {query.strip()}")
    
    # LangGraph agents use messages format
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    
    print("\n🧠 ML Agent Response:")
    # Extract final answer from messages
    if "messages" in result:
        print(result["messages"][-1].content)
    else:
        # Fallback for legacy API
        print(result.get("output", str(result)))


def example_3_structured_output():
    """Example 3: Get typed Python objects instead of text."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Structured Output (Typed Objects)")
    print("=" * 60)
    
    from polymarket_agents.langchain.agent import (
        create_structured_probability_agent,
        MarketForecast
    )
    from langchain_core.prompts import ChatPromptTemplate
    
    # This agent returns MarketForecast objects, not prose
    llm = create_structured_probability_agent(model="gpt-4o-mini")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a probability forecaster. Analyze the market and return a structured forecast."),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    
    # Query
    result = chain.invoke({
        "query": "Will Bitcoin hit $100k by end of 2026? Current BTC price is ~$105k. Market shows 65% probability."
    })
    
    # Result is a typed MarketForecast object!
    print("\n📈 Structured Forecast:")
    print(f"  Probability: {result.probability:.1%}")
    print(f"  Confidence:  {result.confidence}")
    print(f"  Reasoning:   {result.reasoning}")
    print(f"  Sources:     {', '.join(result.data_sources)}")


def example_4_domain_agent():
    """Example 4: Use domain-specific crypto agent."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Domain-Specific Agent (Crypto)")
    print("=" * 60)
    
    from polymarket_agents.domains.registry import get_domain, list_domains
    from polymarket_agents.context import get_context
    
    # List available domains
    print(f"\n🌐 Available domains: {list_domains()}")
    
    # Get crypto domain
    crypto_domain = get_domain("crypto")
    
    if crypto_domain:
        print("\n🔍 Running crypto scanner...")
        print("   (This scans Polymarket for BTC/ETH markets, enriches with live prices,")
        print("    calculates edge, and recommends trades using Kelly criterion)")
        
        # Create agent
        crypto_agent = crypto_domain.create_agent(get_context())
        
        # Run analysis (may take 30-60 seconds)
        try:
            recommendations = crypto_agent.run()
            
            print(f"\n✅ Found {len(recommendations)} opportunities:")
            
            for i, rec in enumerate(recommendations[:3], 1):  # Show top 3
                print(f"\n{i}. {rec.market.question}")
                print(f"   Market Price: {rec.edge.market_prob:.1%}")
                print(f"   Our Estimate: {rec.edge.our_prob:.1%}")
                print(f"   Edge:         {rec.edge.edge:+.1%}")
                print(f"   Action:       {rec.action}")
                print(f"   Position:     {rec.size_fraction:.1%} of bankroll")
                print(f"   Reasoning:    {rec.reasoning[:100]}...")
        except Exception as e:
            print(f"\n⚠️  Error running crypto scanner: {e}")
            print("   (This may fail if no crypto markets are active or API is down)")
    else:
        print("⚠️  Crypto domain not available")


def example_5_tool_calling_trace():
    """Example 5: See the ReAct reasoning loop."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: ReAct Loop (Reasoning Trace)")
    print("=" * 60)
    
    from polymarket_agents.langchain.agent import create_polymarket_agent
    from langchain_core.messages import HumanMessage
    
    # verbose=True shows the agent's thought process
    agent = create_polymarket_agent(
        model="gpt-4o-mini",
        verbose=True,  # ← This shows tool calls!
        max_iterations=5
    )
    
    print("\n🔍 Asking agent to find profitable markets...")
    print("   (Watch the tool calls below)\n")
    
    query = "Find a crypto market with volume >$5k and tell me if it's mispriced"
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    
    print("\n" + "=" * 60)
    print("Final Answer:")
    # Extract final answer from messages
    if "messages" in result:
        print(result["messages"][-1].content)
    else:
        # Fallback for legacy API
        print(result.get("output", str(result)))


def example_6_custom_tool():
    """Example 6: Add your own custom tool."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Custom Tool (Sentiment Analysis)")
    print("=" * 60)
    
    from polymarket_agents.langchain.agent import create_polymarket_agent
    from polymarket_agents.tooling import wrap_tool
    from langchain_core.messages import HumanMessage
    from pydantic import BaseModel, Field
    
    # Define a simple sentiment tool
    def analyze_sentiment(text: str) -> str:
        """Analyze sentiment of text (dummy implementation)."""
        positive_words = ["win", "succeed", "beat", "above", "bullish"]
        negative_words = ["lose", "fail", "below", "bearish"]
        
        text_lower = text.lower()
        pos = sum(1 for w in positive_words if w in text_lower)
        neg = sum(1 for w in negative_words if w in text_lower)
        
        if pos > neg:
            return f'{{"sentiment": "positive", "score": {0.6 + pos * 0.1}}}'
        elif neg > pos:
            return f'{{"sentiment": "negative", "score": {0.4 - neg * 0.1}}}'
        return '{"sentiment": "neutral", "score": 0.5}'
    
    class SentimentInput(BaseModel):
        text: str = Field(description="Text to analyze")
    
    # Wrap as LangChain tool
    sentiment_tool = wrap_tool(
        analyze_sentiment,
        name="analyze_sentiment",
        description="Analyze sentiment of text. Returns JSON with sentiment and score.",
        args_schema=SentimentInput
    )
    
    # Create agent with only this tool
    agent = create_polymarket_agent(
        model="gpt-4o-mini",
        tools=[sentiment_tool],
        max_iterations=3
    )
    
    query = "What's the sentiment of: 'Bitcoin will definitely beat $100k this year!'"
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    
    print("\n💭 Sentiment Analysis:")
    # Extract final answer from messages
    if "messages" in result:
        print(result["messages"][-1].content)
    else:
        # Fallback for legacy API
        print(result.get("output", str(result)))


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("🎓 LangChain Quickstart - Your Production System")
    print("=" * 60)
    print("\nThis shows the LangChain agents you've already built.")
    print("Each example demonstrates a different pattern.\n")
    
    examples = [
        ("1", "Basic agent query", example_1_simple_query),
        ("2", "ML prediction", example_2_ml_prediction),
        ("3", "Structured output", example_3_structured_output),
        ("4", "Domain agent", example_4_domain_agent),
        ("5", "ReAct trace", example_5_tool_calling_trace),
        ("6", "Custom tool", example_6_custom_tool),
    ]
    
    print("Available examples:")
    for num, desc, _ in examples:
        print(f"  {num}. {desc}")
    
    choice = input("\nRun which example? (1-6, or 'all'): ").strip().lower()
    
    if choice == "all":
        for _, _, func in examples:
            try:
                func()
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
    elif choice in [str(i) for i in range(1, 7)]:
        idx = int(choice) - 1
        try:
            examples[idx][2]()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Invalid choice. Run with '1', '2', '3', '4', '5', '6', or 'all'")
    
    print("\n" + "=" * 60)
    print("✅ Quickstart Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Read LANGCHAIN_TUTORIAL.md for deep dive")
    print("  2. Explore langchain/agent.py for more agent types")
    print("  3. Check langchain/tools.py for all 50+ tools")
    print("  4. Try adding your own custom tool")


if __name__ == "__main__":
    main()

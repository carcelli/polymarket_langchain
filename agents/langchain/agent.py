"""
LangChain Agent for Polymarket Trading

This module provides ready-to-use agents for autonomous Polymarket analysis and trading.

Example Usage:
    from agents.langchain.agent import create_polymarket_agent, run_agent

    agent = create_polymarket_agent()
    result = run_agent(agent, "Find the best political market to trade")
    print(result)
"""

import os
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# AGENT CREATION FUNCTIONS
# =============================================================================


from agents.utils.context import ContextManager

def create_polymarket_agent(
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_iterations: int = 10,
    tools: Optional[List] = None,
    verbose: bool = True,
    context_manager: Optional[ContextManager] = None,
):
    """Create a LangChain agent configured for Polymarket analysis.

    Args:
        model: OpenAI model to use.
        temperature: 0.0 = deterministic, 1.0 = creative.
        max_iterations: Maximum tool calls before stopping.
        tools: List of tools to give the agent. If None, uses all available.
        verbose: If True, prints agent reasoning steps.
        context_manager: Optional ContextManager for context engineering.

    Returns:
        AgentExecutor ready to invoke with queries
    """
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_react_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate

    from agents.langchain.tools import get_all_tools

    # Initialize LLM
    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Get tools
    if tools is None:
        tools = get_all_tools()

    # Get context block if manager provided
    context_block = ""
    if context_manager:
        context_block = context_manager.get_model_context()

    # Create prompt template
    system_message = f"""You are an expert Polymarket trader and analyst.

{context_block}

Your role is to:
1. Analyze prediction markets using available tools
2. Find trading opportunities with positive expected value
3. Provide probabilistic forecasts based on evidence
4. Explain your reasoning clearly

IMPORTANT GUIDELINES:
- Always fetch current data before making recommendations
- Consider multiple information sources (news, orderbook, market data)
- Express predictions as probabilities, not certainties
- Note any uncertainties or limitations in your analysis
- For trades, consider risk/reward and position sizing

Available tools: {{tool_names}}

Tool descriptions:
{{tools}}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{{tool_names}}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", "{input}\n\n{agent_scratchpad}"),
        ]
    )

    # Create agent
    agent = create_react_agent(llm, tools, prompt)

    # Wrap in executor
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=verbose,
        max_iterations=max_iterations,
        handle_parsing_errors=True,
    )

    return executor


def create_simple_analyst(model: str = "gpt-4o-mini"):
    """Create a simpler agent focused on market analysis (no trading tools).

    Good for research and recommendations without execution capability.

    Args:
        model: OpenAI model to use

    Returns:
        AgentExecutor for analysis tasks
    """
    from agents.langchain.tools import (
        get_market_tools,
        get_event_tools,
        get_analysis_tools,
    )

    analysis_tools = get_market_tools() + get_event_tools() + get_analysis_tools()
    return create_polymarket_agent(
        model=model,
        tools=analysis_tools,
        max_iterations=8,
    )


def create_research_agent(model: str = "gpt-4o-mini"):
    """Create an agent specialized for market research with news integration.

    Args:
        model: OpenAI model to use

    Returns:
        AgentExecutor for research tasks
    """
    from agents.langchain.tools import (
        fetch_all_markets,
        fetch_all_events,
        search_news,
        get_superforecast,
        query_markets_rag,
    )

    research_tools = [
        fetch_all_markets,
        fetch_all_events,
        search_news,
        get_superforecast,
        query_markets_rag,
    ]

    return create_polymarket_agent(
        model=model,
        tools=research_tools,
        temperature=0.3,  # Slightly more creative for research
    )


# =============================================================================
# AGENT EXECUTION HELPERS
# =============================================================================


def run_agent(agent, query: str) -> str:
    """Run an agent with a query and return the result.

    Args:
        agent: AgentExecutor from create_polymarket_agent()
        query: Natural language query/instruction

    Returns:
        Agent's final answer as string
    """
    try:
        result = agent.invoke({"input": query})
        return result.get("output", str(result))
    except Exception as e:
        return f"Agent error: {str(e)}"


def run_analysis_chain(query: str, model: str = "gpt-4o-mini") -> str:
    """Quick function to run a one-off analysis.

    Creates a temporary agent, runs the query, returns result.
    For repeated use, create an agent once and reuse it.

    Args:
        query: What to analyze
        model: OpenAI model to use

    Returns:
        Analysis result as string
    """
    agent = create_simple_analyst(model)
    return run_agent(agent, query)


# =============================================================================
# SPECIALIZED WORKFLOWS
# =============================================================================


def find_best_trade(
    category: Optional[str] = None,
    risk_tolerance: str = "medium",
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Automated workflow to find the best trade opportunity.

    Args:
        category: Optional category filter (e.g., "politics", "sports")
        risk_tolerance: "low", "medium", or "high"
        model: OpenAI model to use

    Returns:
        Dict with trade recommendation and reasoning
    """
    agent = create_polymarket_agent(model=model)

    query = f"""
    Find the best trading opportunity on Polymarket right now.
    
    Requirements:
    - {"Focus on " + category + " markets" if category else "Any category is fine"}
    - Risk tolerance: {risk_tolerance}
    - Look for mispriced markets where your forecast differs from market price
    - Consider liquidity and spread
    
    Steps:
    1. Fetch current tradeable markets
    2. Analyze the most interesting ones
    3. Get news/research on the top candidates
    4. Use superforecaster methodology to estimate probabilities
    5. Compare your probability to market prices
    6. Recommend a specific trade with reasoning
    
    Return your recommendation with:
    - Market question
    - Current market price
    - Your probability estimate
    - Recommended position (buy/sell, which outcome)
    - Expected value calculation
    - Key risks
    """

    result = run_agent(agent, query)

    return {
        "recommendation": result,
        "category": category,
        "risk_tolerance": risk_tolerance,
        "model": model,
    }


def analyze_specific_market(market_question: str, model: str = "gpt-4o-mini") -> str:
    """Deep dive analysis on a specific market.

    Args:
        market_question: The prediction question to analyze
        model: OpenAI model to use

    Returns:
        Comprehensive analysis
    """
    agent = create_research_agent(model=model)

    query = f"""
    Provide a comprehensive analysis of this prediction market:
    
    "{market_question}"
    
    Include:
    1. Current market price and liquidity
    2. Relevant news and recent developments
    3. Historical context and base rates
    4. Key factors that could move the market
    5. Probability estimate with confidence interval
    6. Trading recommendation (if any edge exists)
    """

    return run_agent(agent, query)


# =============================================================================
# LANGGRAPH MULTI-STEP AGENT (Advanced)
# =============================================================================


def create_langgraph_trader():
    """Create a LangGraph-based multi-step trading agent.

    LangGraph provides more control over multi-step workflows
    compared to basic ReAct agents. Good for complex trading
    strategies with multiple stages.

    Returns:
        LangGraph CompiledGraph

    Note: Requires langgraph package: pip install langgraph
    """
    try:
        from langgraph.graph import StateGraph, END
        from langgraph.prebuilt import ToolNode
        from langchain_openai import ChatOpenAI
        from typing import TypedDict, Annotated, Sequence
        from langchain_core.messages import BaseMessage
        import operator

        from agents.langchain.tools import get_all_tools

        # Define the state
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], operator.add]
            next_action: str

        # Initialize LLM with tools
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        tools = get_all_tools()
        llm_with_tools = llm.bind_tools(tools)

        # Define nodes
        def call_model(state: AgentState):
            messages = state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def should_continue(state: AgentState):
            last_message = state["messages"][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"
            return END

        # Build graph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(tools))

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", END: END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    except ImportError:
        raise ImportError(
            "LangGraph is required for this agent. "
            "Install with: pip install langgraph"
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Polymarket LangChain Agent Examples")
    print("=" * 60)

    # Example 1: Simple analysis
    print("\n1. Creating simple analyst agent...")
    agent = create_simple_analyst()

    # Example 2: Run a query
    print("\n2. Fetching current markets...")
    result = run_agent(
        agent, "What are the top 3 most interesting markets to trade right now?"
    )
    print(result)

    # Example 3: Find best trade (commented out - runs full workflow)
    # print("\n3. Finding best trade...")
    # trade = find_best_trade(category="politics", risk_tolerance="medium")
    # print(trade["recommendation"])

"""
LangChain Agent for Polymarket Trading

This module provides ready-to-use agents for autonomous Polymarket analysis and trading.

Example Usage:
    from polymarket_agents.langchain.agent import create_polymarket_agent, run_agent

    agent = create_polymarket_agent()
    result = run_agent(agent, "Find the best political market to trade")
    print(result)
"""

import os
import logging
from typing import Optional, List, Dict, Any, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field

BaseModelV1 = BaseModel
FieldV1 = Field

from polymarket_agents.utils.context import ContextManager
from polymarket_agents.config import DEFAULT_MODEL

load_dotenv()

# =============================================================================
# AGENT CREATION FUNCTIONS
# =============================================================================


def create_polymarket_agent(
    llm: Optional["BaseChatModel"] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_iterations: int = 10,
    tools: Optional[List] = None,
    verbose: bool = True,
    context_manager: Optional[ContextManager] = None,
    **llm_kwargs,
):
    """Create a LangChain agent configured for Polymarket analysis.

    Args:
        llm: Optional pre-configured language model. If None, creates ChatOpenAI.
        model: OpenAI model to use (ignored if llm provided).
        temperature: 0.0 = deterministic, 1.0 = creative.
        max_iterations: Maximum tool calls before stopping.
        tools: List of tools to give the agent. If None, uses all available.
        verbose: If True, prints agent reasoning steps.
        context_manager: Optional ContextManager for context engineering.
        **llm_kwargs: Additional kwargs passed to LLM constructor.

    Returns:
        AgentExecutor ready to invoke with queries
    """
    try:
        # Try modern LangGraph API first (better JSON parsing)
        from langgraph.prebuilt import create_react_agent
        from langchain_core.language_models import BaseChatModel

        # Initialize LLM
        if llm is None:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
                **llm_kwargs,
            )
        elif not isinstance(llm, BaseChatModel):
            raise ValueError("llm must be an instance of BaseChatModel")

        # Get tools
        if tools is None:
            from polymarket_agents.langchain.tools import get_all_tools

            tools = get_all_tools()

        # Get context block if manager provided
        context_block = ""
        if context_manager:
            context_block = context_manager.get_model_context()

        # Create system prompt
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
- For trades, consider risk/reward and position sizing"""

        # LangGraph's create_react_agent handles JSON parsing correctly
        agent = create_react_agent(llm, tools, state_modifier=system_message)

        return agent

    except ImportError:
        # Fallback to legacy LangChain if LangGraph not available
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.language_models import BaseChatModel

        from polymarket_agents.langchain.tools import get_all_tools

        # Initialize LLM
        if llm is None:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=os.getenv("OPENAI_API_KEY"),
                **llm_kwargs,
            )
        elif not isinstance(llm, BaseChatModel):
            raise ValueError("llm must be an instance of BaseChatModel")

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


def create_simple_analyst(
    llm: Optional["BaseChatModel"] = None, model: str = DEFAULT_MODEL, **llm_kwargs
):
    """Create a simpler agent focused on market analysis (no trading tools).

    Good for research and recommendations without execution capability.

    Args:
        model: OpenAI model to use

    Returns:
        AgentExecutor for analysis tasks
    """
    from polymarket_agents.langchain.tools import (
        get_market_tools,
        get_event_tools,
        get_analysis_tools,
    )

    analysis_tools = get_market_tools() + get_event_tools() + get_analysis_tools()
    return create_polymarket_agent(
        llm=llm, model=model, tools=analysis_tools, max_iterations=8, **llm_kwargs
    )


def create_structured_probability_agent(
    llm: Optional["BaseChatModel"] = None, model: str = "gpt-4o-mini", **llm_kwargs
) -> "BaseChatModel":
    """Create a structured probability extraction agent that returns typed MarketForecast objects.

    This agent is designed for programmatic consumption - it returns structured data
    instead of prose, making it suitable for dashboards, automated trading systems,
    and ML pipeline integration.

    Args:
        llm: Optional pre-configured language model
        model: OpenAI model to use (ignored if llm provided)
        **llm_kwargs: Additional LLM parameters

    Returns:
        LLM instance configured for structured output
    """
    if llm is None:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            temperature=0,  # Deterministic for structured output
            **llm_kwargs,
        )

    # Bind the structured output schema to the LLM
    structured_llm = llm.with_structured_output(MarketForecast, method="json_mode")

    return structured_llm


def create_probability_extraction_agent(
    llm: Optional["BaseChatModel"] = None, model: str = DEFAULT_MODEL, **llm_kwargs
):
    """Create an agent specialized for extracting implied probabilities from Polymarket data.

    Focuses on converting market prices to probabilities and analyzing crowd wisdom
    for business forecasting applications (recessions, elections, economic events).

    Args:
        model: OpenAI model to use

    Returns:
        AgentExecutor for probability extraction tasks
    """
    from polymarket_agents.langchain.tools import (
        get_current_markets_gamma,
        get_market_by_id,
        get_superforecast,
        analyze_market_with_llm,
        get_top_volume_markets,
        search_markets_db,
        get_database_stats,
    )

    probability_tools = [
        get_current_markets_gamma,
        get_market_by_id,
        get_superforecast,
        analyze_market_with_llm,
        get_top_volume_markets,
        search_markets_db,
        get_database_stats,
    ]

    return create_polymarket_agent(
        llm=llm,
        model=model,
        tools=probability_tools,
        temperature=0.0,  # Deterministic for probability extraction
        max_iterations=8,
        **llm_kwargs,
    )


def create_research_agent(
    llm: Optional["BaseChatModel"] = None, model: str = DEFAULT_MODEL, **llm_kwargs
):
    """Create an agent specialized for market research with news integration.

    Args:
        model: OpenAI model to use

    Returns:
        AgentExecutor for research tasks
    """
    from polymarket_agents.langchain.tools import (
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
        llm=llm,
        model=model,
        tools=research_tools,
        temperature=0.3,  # Slightly more creative for research
        **llm_kwargs,
    )


# =============================================================================
# AGENT EXECUTION HELPERS
# =============================================================================

logger = logging.getLogger(__name__)


class AgentExecutionError(Exception):
    """Custom exception for agent execution failures."""

    pass


def run_agent(agent, query: str) -> str:
    """Run an agent with a query and return the result.

    Args:
        agent: AgentExecutor from create_polymarket_agent()
        query: Natural language query/instruction

    Returns:
        Agent's final answer as string

    Raises:
        AgentExecutionError: When agent execution fails critically
    """
    try:
        logger.info(f"Invoking agent with query: {query[:50]}...")
        result = agent.invoke({"input": query})

        output = result.get("output")
        if not output:
            # Sometimes agents finish but return empty output
            raise AgentExecutionError("Agent finished but returned no output.")

        return str(output)

    except Exception as e:
        logger.error(f"Agent failed: {e}", exc_info=True)
        # Re-raise so the caller knows it failed.
        # Don't return a string containing "error"!
        raise AgentExecutionError(f"Critical failure in agent execution: {e}") from e


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


def create_ml_forecast_comparison_agent(
    llm: Optional["BaseChatModel"] = None, model: str = DEFAULT_MODEL, **llm_kwargs
):
    """Create an agent that compares Polymarket crowd wisdom against ML model forecasts.

    Useful for small businesses to validate their internal forecasting models
    against market-based probabilities for events like recessions, elections, etc.

    Args:
        model: OpenAI model to use

    Returns:
        AgentExecutor for ML comparison tasks
    """
    from polymarket_agents.langchain.tools import (
        get_current_markets_gamma,
        get_superforecast,
        analyze_market_with_llm,
        search_markets_db,
        get_top_volume_markets,
        get_market_from_db,
        list_recent_markets,
    )

    comparison_tools = [
        get_current_markets_gamma,
        get_superforecast,
        analyze_market_with_llm,
        search_markets_db,
        get_top_volume_markets,
        get_market_from_db,
        list_recent_markets,
    ]

    return create_polymarket_agent(
        llm=llm,
        model=model,
        tools=comparison_tools,
        temperature=0.2,  # Balanced creativity for analysis
        max_iterations=12,  # More iterations for complex comparisons
        **llm_kwargs,
    )


class ForecastComparison(BaseModel):
    """Structured output for ML vs market forecast comparison."""

    event_description: str = Field(description="The event being forecasted")
    ml_forecast_probability: float = Field(
        description="Your ML model's probability estimate (0.0-1.0)"
    )
    market_consensus_probability: Optional[float] = Field(
        description="Polymarket crowd consensus probability"
    )
    difference: Optional[float] = Field(
        description="Difference between ML and market (ML - Market)"
    )
    confidence_assessment: str = Field(
        description="Assessment of which forecast to trust and why"
    )
    business_implications: str = Field(
        description="Business implications and recommended actions"
    )
    key_risks: str = Field(description="Key risks and uncertainties to consider")


class MarketForecast(BaseModelV1):
    """Structured output for market probability analysis."""

    market_question: str = FieldV1(description="The market question being analyzed")
    implied_probability: float = FieldV1(
        description="0.0 to 1.0 probability implied by market prices"
    )
    confidence_score: float = FieldV1(
        description="0.0 to 1.0 confidence in the analysis"
    )
    reasoning_summary: str = FieldV1(description="Brief summary of the reasoning")
    recommended_action: str = FieldV1(description="BUY, SELL, or HOLD recommendation")
    key_factors: list[str] = FieldV1(
        description="Key factors influencing the probability"
    )


def compare_ml_vs_market_forecast(
    ml_forecast: float,
    event_description: str,
    llm: Optional["BaseChatModel"] = None,
    model: str = DEFAULT_MODEL,
    structured_output: bool = True,
    **llm_kwargs,
) -> Union[Dict[str, Any], ForecastComparison]:
    """Compare your ML model's forecast against Polymarket crowd wisdom.

    Args:
        ml_forecast: Your ML model's probability estimate (0.0 to 1.0)
        event_description: Description of the event being forecasted
        llm: Optional pre-configured language model
        model: OpenAI model to use (ignored if llm provided)
        structured_output: If True, returns typed ForecastComparison object
        **llm_kwargs: Additional LLM parameters

    Returns:
        Dict with comparison analysis or structured ForecastComparison object
    """
    if structured_output:
        # Use structured output with Pydantic
        agent = create_ml_forecast_comparison_agent(llm=llm, model=model, **llm_kwargs)

        # Create a structured prompt for better parsing
        structured_prompt = f"""
        Compare my ML model's forecast against Polymarket crowd wisdom for this event.

        Event: {event_description}
        My ML Forecast: {ml_forecast:.3f} (probability between 0.0 and 1.0)

        You must respond with a JSON object containing exactly these fields:
        - market_consensus_probability: The Polymarket crowd consensus probability (0.0-1.0) or null if not found
        - difference: The difference (my_forecast - market_consensus) or null
        - confidence_assessment: Brief assessment of which forecast to trust
        - business_implications: Key business implications and recommendations
        - key_risks: Important risks and uncertainties to consider

        Be precise and data-driven in your analysis.
        """

        result = run_agent(agent, structured_prompt)

        # Try to parse structured output, fall back to dict if parsing fails
        try:
            # Extract JSON from the result (agents often wrap JSON in text)
            import json
            import re

            # Look for JSON-like content in the result
            json_match = re.search(r"\{.*\}", result, re.DOTALL)
            if json_match:
                parsed_data = json.loads(json_match.group())
                return ForecastComparison(
                    event_description=event_description,
                    ml_forecast_probability=ml_forecast,
                    market_consensus_probability=parsed_data.get(
                        "market_consensus_probability"
                    ),
                    difference=parsed_data.get("difference"),
                    confidence_assessment=parsed_data.get(
                        "confidence_assessment", "Unable to assess"
                    ),
                    business_implications=parsed_data.get(
                        "business_implications", "No specific implications identified"
                    ),
                    key_risks=parsed_data.get(
                        "key_risks", "Standard market risks apply"
                    ),
                )
        except (json.JSONDecodeError, KeyError, TypeError):
            # Fall back to dict format
            pass

    # Original dict-based implementation
    agent = create_ml_forecast_comparison_agent(llm=llm, model=model, **llm_kwargs)

    query = f"""
    Compare my ML model's forecast against Polymarket crowd wisdom for this event:

    Event: {event_description}
    My ML Forecast: {ml_forecast:.1%} probability

    Steps to follow:
    1. Search for relevant Polymarket markets matching this event
    2. Get current market prices and implied probabilities
    3. Analyze the difference between my forecast and market consensus
    4. Consider reasons for any discrepancies (market efficiency, information asymmetry, etc.)
    5. Provide business insights: Should I trust my model or the market?

    Return analysis with:
    - Market consensus probability
    - Forecast difference (my model vs market)
    - Confidence assessment for both approaches
    - Business implications and recommended actions
    """

    result = run_agent(agent, query)

    if structured_output:
        # Return structured object even if parsing failed
        return ForecastComparison(
            event_description=event_description,
            ml_forecast_probability=ml_forecast,
            market_consensus_probability=None,
            difference=None,
            confidence_assessment="Analysis completed but structured parsing failed",
            business_implications=result,
            key_risks="See full analysis for risk details",
        )

    return {
        "comparison": result,
        "ml_forecast": ml_forecast,
        "event": event_description,
        "model": model,
    }


def extract_market_probability(
    market_question: str,
    llm: Optional["BaseChatModel"] = None,
    model: str = "gpt-4o-2024-08-06",
    **llm_kwargs,
) -> MarketForecast:
    """Extract structured probability analysis for a specific market question.

    Returns typed MarketForecast object instead of prose, suitable for:
    - Dashboard displays
    - Automated trading decisions
    - ML pipeline integration
    - Database storage

    Args:
        market_question: The market question to analyze
        llm: Optional pre-configured language model
        model: OpenAI model to use (ignored if llm provided)
        **llm_kwargs: Additional LLM parameters

    Returns:
        MarketForecast object with structured analysis

    Raises:
        AgentExecutionError: If analysis fails
    """
    try:
        structured_llm = create_structured_probability_agent(
            llm=llm, model=model, **llm_kwargs
        )

        prompt = f"""
        Analyze this prediction market and provide a structured probability assessment.

        Market Question: {market_question}

        You must search for relevant Polymarket data and provide analysis in the exact JSON format specified.
        Focus on current market prices, trading volume, and crowd sentiment.
        """

        logger.info(f"Extracting probability for: {market_question[:50]}...")
        result = structured_llm.invoke(prompt)

        if not isinstance(result, MarketForecast):
            raise AgentExecutionError(f"Expected MarketForecast, got {type(result)}")

        return result

    except Exception as e:
        logger.error(f"Probability extraction failed: {e}", exc_info=True)
        raise AgentExecutionError(f"Failed to extract market probability: {e}") from e


def analyze_specific_market(
    market_question: str,
    llm: Optional["BaseChatModel"] = None,
    model: str = DEFAULT_MODEL,
    **llm_kwargs,
) -> str:
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


def create_crypto_agent(
    llm: Optional["BaseChatModel"] = None,
    model: str = DEFAULT_MODEL,
    risk_tolerance: str = "medium",
    **llm_kwargs,
):
    """Create a crypto-focused agent for cryptocurrency markets and predictions.

    Specialized for analyzing Bitcoin, Ethereum, and crypto market events.
    Perfect for crypto traders and businesses with crypto exposure.

    Args:
        llm: Optional pre-configured language model
        model: OpenAI model to use (ignored if llm provided)
        risk_tolerance: "low", "medium", or "high"
        **llm_kwargs: Additional LLM parameters

    Returns:
        AgentExecutor configured for crypto analysis
    """
    return create_business_domain_agent(
        domain="crypto",
        llm=llm,
        model=model,
        risk_tolerance=risk_tolerance,
        **llm_kwargs,
    )


def create_sports_agent(
    llm: Optional["BaseChatModel"] = None,
    model: str = DEFAULT_MODEL,
    risk_tolerance: str = "medium",
    **llm_kwargs,
):
    """Create a sports-focused agent for sports betting and outcomes.

    Specialized for major sports events, championships, and tournament predictions.
    Great for sports analytics and betting businesses.

    Args:
        llm: Optional pre-configured language model
        model: OpenAI model to use (ignored if llm provided)
        risk_tolerance: "low", "medium", or "high"
        **llm_kwargs: Additional LLM parameters

    Returns:
        AgentExecutor configured for sports analysis
    """
    return create_business_domain_agent(
        domain="sports",
        llm=llm,
        model=model,
        risk_tolerance=risk_tolerance,
        **llm_kwargs,
    )


def create_business_domain_agent(
    domain: str = "crypto",
    llm: Optional["BaseChatModel"] = None,
    model: str = DEFAULT_MODEL,
    risk_tolerance: str = "medium",
    **llm_kwargs,
):
    """Create a domain-specialized agent for business-relevant market analysis.

    Args:
        domain: "economy", "politics", "crypto", "sports", or "general"
        model: OpenAI model to use
        risk_tolerance: "low", "medium", or "high"

    Returns:
        AgentExecutor configured for the specific business domain
    """
    from polymarket_agents.langchain.tools import (
        get_markets_by_category,
        get_top_volume_markets,
        search_markets_db,
        get_superforecast,
        analyze_market_with_llm,
        search_news,
    )

    # Domain-specific configurations (focusing on crypto and sports)
    domain_configs = {
        "crypto": {
            "categories": ["crypto"],
            "focus": "cryptocurrency prices, adoption, regulation, market predictions",
            "keywords": [
                "bitcoin",
                "ethereum",
                "crypto",
                "blockchain",
                "regulation",
                "defi",
                "nft",
            ],
        },
        "sports": {
            "categories": ["sports"],
            "focus": "sports outcomes, championships, player performance, tournament results",
            "keywords": [
                "championship",
                "season",
                "playoffs",
                "tournament",
                "super bowl",
                "world series",
                "finals",
            ],
        },
        "economy": {
            "categories": ["economy", "finance"],
            "focus": "economic indicators, recessions, inflation, GDP growth",
            "keywords": [
                "recession",
                "inflation",
                "GDP",
                "Fed",
                "economy",
                "unemployment",
            ],
        },
        "politics": {
            "categories": ["politics"],
            "focus": "elections, policy changes, political events",
            "keywords": ["election", "president", "congress", "policy", "political"],
        },
    }

    domain_configs.get(
        domain,
        {
            "categories": None,
            "focus": f"{domain} events and outcomes",
            "keywords": [domain],
        },
    )

    domain_tools = [
        get_markets_by_category,
        get_top_volume_markets,
        search_markets_db,
        get_superforecast,
        analyze_market_with_llm,
        search_news,
    ]

    agent = create_polymarket_agent(
        llm=llm, model=model, tools=domain_tools, temperature=0.1, **llm_kwargs
    )

    return agent


def analyze_business_risks(
    business_type: str,
    domain: str = "crypto",
    llm: Optional["BaseChatModel"] = None,
    model: str = DEFAULT_MODEL,
    **llm_kwargs,
) -> str:
    """Analyze business risks using Polymarket data for a specific business type.

    Args:
        business_type: Description of your business (e.g., "crypto trading firm")
        domain: Business domain to focus on ("crypto", "sports", "economy", "politics")
        model: OpenAI model to use

    Returns:
        Risk analysis and recommendations
    """
    agent = create_business_domain_agent(domain=domain, model=model)

    query = f"""
    Analyze business risks for a {business_type} using Polymarket prediction markets.

    Business Type: {business_type}
    Risk Domain Focus: {domain}

    Provide:
    1. Current market probabilities for relevant {domain} events
    2. Risk assessment: Which events pose the biggest threats/opportunities?
    3. Probability-weighted impact analysis
    4. Recommended risk mitigation strategies
    5. Market-based contingency planning advice

    Focus on actionable insights for small business owners.
    """

    return run_agent(agent, query)


def create_langgraph_trader(
    llm: Optional["BaseChatModel"] = None, tools: Optional[List] = None
):
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

        from polymarket_agents.langchain.tools import get_all_tools

        # Define the state
        class AgentState(TypedDict):
            messages: Annotated[Sequence[BaseMessage], operator.add]
            next_action: str

        # Use passed LLM and tools, or defaults
        if llm is None:
            from langchain_openai import ChatOpenAI

            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

        if tools is None:
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


def demo_probability_extraction():
    """Demo the probability extraction agent."""
    print("\n" + "=" * 60)
    print("PROBABILITY EXTRACTION AGENT DEMO")
    print("=" * 60)

    agent = create_probability_extraction_agent()

    query = """
    Extract implied probabilities from current Polymarket data for major 2026 US economic events.
    Focus on recession risks, Fed policy, and GDP growth. Summarize the top 5 markets with
    highest volume and provide business-relevant insights for small business planning.
    """

    print("Query:", query.strip())
    result = run_agent(agent, query)
    print("\nResult:")
    print(result)
    return result


def demo_ml_comparison():
    """Demo comparing ML forecast against market consensus."""
    print("\n" + "=" * 60)
    print("ML FORECAST COMPARISON DEMO")
    print("=" * 60)

    # Example: Your ML model predicts 35% recession probability
    ml_forecast = 0.35
    event = "US recession in 2026"

    print(f"Your ML Forecast: {ml_forecast:.1%} probability of {event}")

    comparison = compare_ml_vs_market_forecast(
        ml_forecast=ml_forecast, event_description=event
    )

    print("\nComparison Analysis:")
    print(comparison["comparison"])
    return comparison


def demo_business_risks():
    """Demo business risk analysis using domain-specific agents."""
    print("\n" + "=" * 60)
    print("BUSINESS RISK ANALYSIS DEMO")
    print("=" * 60)

    business_type = "small retail business with physical locations"
    domain = "economy"

    print(f"Business Type: {business_type}")
    print(f"Risk Domain: {domain}")

    analysis = analyze_business_risks(business_type=business_type, domain=domain)

    print("\nRisk Analysis:")
    print(analysis)
    return analysis


if __name__ == "__main__":
    print("=" * 60)
    print("Polymarket LangChain Agent Examples")
    print("=" * 60)

    try:
        # Demo 1: Probability extraction for business forecasting
        demo_probability_extraction()

        # Demo 2: ML forecast comparison
        demo_ml_comparison()

        # Demo 3: Business risk analysis
        demo_business_risks()

        print("\n" + "=" * 60)
        print("TARGET EVENTS FOR TESTING:")
        print("=" * 60)
        print("• US 2026 Midterm Elections")
        print("• Federal Reserve Interest Rate Decisions")
        print("• Q4 2026 GDP Growth Forecasts")
        print("• Bitcoin ETF Approval Outcomes")
        print("• Major Tech Company Earnings (Meta, Google, Amazon)")
        print("• Global Climate Agreement Progress")
        print("• US-China Trade Relations")

    except Exception as e:
        print(f"Demo error: {e}")
        print(
            "Make sure you have OPENAI_API_KEY set and required dependencies installed."
        )

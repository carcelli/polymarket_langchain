# agent_with_models.py
import os
import sys
import json
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, ValidationError

# Add project root to path
sys.path.append(str(Path(__file__).parents[1] / "src"))

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Import existing project models and clients
from polymarket_agents.utils.objects import Market, PolymarketEvent
from polymarket_agents.connectors.gamma import GammaMarketClient

# --- 1. Ontology / Data Contracts ---
# Integrating Pydantic models as the "Metaphysical" layer for the agent.
# These ensure that data flowing between tools and the agent is structured and valid.

class MarketForecast(BaseModel):
    """Output model for the ML forecasting tool."""
    market_id: str
    predicted_volume_trend: str = Field(..., description="Up/Down/Flat")
    confidence_score: float = Field(..., ge=0, le=1)
    business_implication: str

# --- 2. Tools ---

@tool
def fetch_polymarket_data(query: str) -> str:
    """
    Fetches and validates Polymarket data for a given search query or market ID.
    Returns a string representation of the validated Pydantic models.
    """
    print(f"\n[Tool] Fetching data for: {query}")
    client = GammaMarketClient()
    
    try:
        # Attempt to interpret query as ID first, then search
        if query.isdigit():
             markets = client.get_markets(querystring_params={"id": query}, parse_pydantic=True)
        else:
            # Search by string (naive search via active markets)
            # Use get_markets directly to ensure we can pass parse_pydantic=True
            params = {
                "active": True,
                "closed": False,
                "archived": False,
                "limit": 10  # Fetch more to filter locally
            }
            markets = client.get_markets(querystring_params=params, parse_pydantic=True)
            
            # Remove quotes if present in query
            clean_query = query.replace('"', '').replace("'", "")
            
            if clean_query.lower() not in ["active", "current markets"]:
                 markets = [m for m in markets if clean_query.lower() in m.question.lower()]
        
        if not markets:
            return "No markets found matching the query."

        # Serialize the first few results
        results_str = ""
        for m in markets[:3]:
             results_str += f"Market ID: {m.id}, Question: {m.question}, Spread: {m.spread}, Volume: {m.volume}\n"
        
        return results_str

    except Exception as e:
        return f"Error fetching data: {str(e)}"

@tool
def analyze_market_opportunity(market_id: str, current_spread: float) -> str:
    """
    Simulates an ML analysis (e.g., PyTorch forecasting) on a specific market.
    """
    print(f"\n[Tool] Analyzing market opportunity for ID: {market_id}")
    
    # Mock inference
    is_good_opportunity = current_spread > 0.01  # Arbitrary logic
    
    forecast = MarketForecast(
        market_id=str(market_id),
        predicted_volume_trend="Up" if is_good_opportunity else "Flat",
        confidence_score=0.85 if is_good_opportunity else 0.4,
        business_implication="High potential for arbitrage or hedging." if is_good_opportunity else "Low actionable insight."
    )
    
    return f"ML Analysis Result: {forecast.json()}"

# --- 3. Agent Construction ---

def create_agent_with_models():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment.")
        return None

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    template = """You are a Data-Driven Business Analyst Agent.
    
    Your goal is to find actionable insights from prediction markets using your tools.
    1. Fetch market data using 'fetch_polymarket_data'.
    2. Analyze the specific market using 'analyze_market_opportunity'.
    3. Synthesize a business recommendation based on the ML output.

    Ontology:
    - Markets have 'spread' (liquidity metric) and 'volume'.
    - High spread/volume correlations indicate opportunities.

    Tools: {tools}
    Tool Names: {tool_names}

    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action, strictly as a valid JSON object.
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    prompt = PromptTemplate.from_template(template)
    
    tools = [fetch_polymarket_data, analyze_market_opportunity]
    
    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 4. Execution ---

if __name__ == "__main__":
    print("Initializing Agent with Pydantic Models Integration...")
    executor = create_agent_with_models()
    
    if executor:
        # Example query
        query = "Find a market about 'MicroStrategy' and analyze it."
        print(f"\n--- Running Agent for: {query} ---\
")
        try:
            result = executor.invoke({"input": query})
            print("\n--- Final Agent Output ---\
")
            print(result['output'])
        except Exception as e:
             print(f"Agent execution failed: {e}")

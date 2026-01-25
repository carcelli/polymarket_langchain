"""
ML Strategy Research Agent

This agent is responsible for autonomously researching, generating, and improving
machine learning strategies for Polymarket trading.

It can:
1. Analyze existing strategies in the codebase.
2. Research new ML techniques via web search.
3. Write new strategy code to the filesystem.
4. Verify that new strategies import and run correctly.
"""

import os
import sys
import glob
import importlib.util
from typing import List, Dict, Any, Optional

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from polymarket_agents.langchain.agent import create_polymarket_agent
from polymarket_agents.tools.research_tools import web_search

# =============================================================================
# FILE SYSTEM TOOLS
# =============================================================================


@tool
def list_strategies() -> str:
    """List all existing ML strategies in the agents/ml_strategies directory."""
    try:
        strategies = glob.glob("agents/ml_strategies/*.py")
        return "\n".join(strategies)
    except Exception as e:
        return f"Error listing strategies: {e}"


@tool
def read_strategy_code(filename: str) -> str:
    """Read the code of a specific strategy file.

    Args:
        filename: Path to the python file (e.g., 'agents/ml_strategies/base_strategy.py')
    """
    try:
        with open(filename, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file {filename}: {e}"


@tool
def write_strategy_code(filename: str, code: str) -> str:
    """Write new strategy code to a file.

    Args:
        filename: Path where to save the file (e.g., 'agents/ml_strategies/new_lstm_strategy.py')
        code: The complete python code for the strategy
    """
    try:
        # Ensure we only write to the ml_strategies directory for safety
        if "agents/ml_strategies/" not in filename:
            return "Error: Can only write to agents/ml_strategies/ directory"

        with open(filename, "w") as f:
            f.write(code)
        return f"Successfully wrote code to {filename}"
    except Exception as e:
        return f"Error writing file {filename}: {e}"


@tool
def verify_strategy(filename: str) -> str:
    """Verify that a strategy file imports correctly and follows the interface.

    Args:
        filename: Path to the strategy file
    """
    try:
        # Construct module name from filename
        module_name = filename.replace("/", ".").replace(".py", "")

        # Dynamic import
        spec = importlib.util.spec_from_file_location(module_name, filename)
        if spec is None:
            return f"Could not load spec for {filename}"

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Check for MLBettingStrategy subclass
        import inspect
        from polymarket_agents.ml_strategies.base_strategy import MLBettingStrategy

        found_strategy = False
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, MLBettingStrategy)
                and obj is not MLBettingStrategy
            ):
                found_strategy = True
                break

        if found_strategy:
            return f"✅ Verification SUCCESS: {filename} contains a valid MLBettingStrategy subclass."
        else:
            return f"❌ Verification FAILED: No subclass of MLBettingStrategy found in {filename}."

    except Exception as e:
        return f"❌ Verification CRASHED: {str(e)}"


# =============================================================================
# RESEARCH AGENT
# =============================================================================


class MLResearchAgent:
    def __init__(self, model: str = "gpt-4o"):
        self.tools = [
            list_strategies,
            read_strategy_code,
            write_strategy_code,
            verify_strategy,
            web_search,
        ]

        self.agent = create_polymarket_agent(
            model=model,
            tools=self.tools,
            temperature=0.2,  # Low temp for coding
            max_iterations=20,  # Allow deeper research/coding loops
        )

    def run_research_cycle(self, focus_area: str = "general") -> str:
        """Run a full research cycle to improve strategies.

        Args:
            focus_area: Specific area to research (e.g., 'time series', 'nlp', 'ensemble')
        """
        prompt = f"""
        You are an expert ML Research Scientist tasked with improving our betting strategies.
        
        Your goal is to CREATE A NEW, WORKING strategy file in 'agents/ml_strategies/'.
        
        Workflow:
        1. LIST existing strategies to understand what we have.
        2. READ the 'base_strategy.py' to understand the interface and 'market_prediction.py' for an example.
        3. RESEARCH new techniques for {focus_area} prediction in financial/betting markets using web_search.
           - Look for 'state of the art time series forecasting' or 'prediction market modeling'.
        4. SELECT one promising technique (e.g., XGBoost, LSTM, Transformer, specialized feature engineering).
        5. WRITE a new strategy file (e.g., 'agents/ml_strategies/xgboost_strategy.py').
           - It MUST inherit from MLBettingStrategy.
           - It MUST implement train(), predict(), and get_feature_importance().
           - It should use standard libraries (sklearn, xgboost, etc.) if possible.
        6. VERIFY the new file using verify_strategy().
        
        Report your findings and the status of the new strategy.
        """

        response = self.agent.invoke({"input": prompt})
        return response.get("output", "No output generated")


if __name__ == "__main__":
    # Example usage
    agent = MLResearchAgent()
    print("Starting ML Research Cycle...")
    result = agent.run_research_cycle(focus_area="gradient boosting")
    print("\n\nResearch Report:\n", result)

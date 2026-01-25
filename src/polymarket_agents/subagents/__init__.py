"""
Subagents for Polymarket Trading System

Specialized subagents that handle specific aspects of market analysis
and trading to keep the main agent context clean and focused.
"""

from .market_research import create_market_research_subagent
from .risk_analysis import create_risk_analysis_subagent
from .strategy_dev import create_strategy_dev_subagent
from .performance_monitor import create_performance_monitor_subagent
from .data_collection import create_data_collection_subagent
from .github_agent import create_github_subagent

__all__ = [
    "create_market_research_subagent",
    "create_risk_analysis_subagent",
    "create_strategy_dev_subagent",
    "create_performance_monitor_subagent",
    "create_data_collection_subagent",
    "create_github_subagent",
    "get_all_subagents",
]


def get_all_subagents():
    """
    Get all specialized subagents for the main trading agent.

    Returns:
        List of subagent configurations ready for deepagents
    """
    return [
        create_market_research_subagent(),
        create_risk_analysis_subagent(),
        create_strategy_dev_subagent(),
        create_performance_monitor_subagent(),
        create_data_collection_subagent(),
        create_github_subagent(),
    ]

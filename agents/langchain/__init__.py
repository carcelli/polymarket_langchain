"""
LangChain Integration Module for Polymarket Agents

This module provides LangChain-compatible tools for interacting with Polymarket.
Use these tools with LangChain agents for autonomous trading decisions.

Two sets of tools are available:
1. Agent Tools (tools.py) - Wrap the polymarket-agents repo functionality
2. CLOB Tools (clob_tools.py) - Wrap the official py_clob_client library

Example Usage:
    # Agent tools (existing polymarket-agents functionality)
    from agents.langchain.tools import get_all_tools
    
    # CLOB tools (official py_clob_client)
    from agents.langchain.clob_tools import get_all_clob_tools
    
    # Combine both for full functionality
    from agents.langchain import get_combined_tools
"""

# Agent tools (polymarket-agents repo)
from agents.langchain.tools import (
    get_all_tools,
    get_market_tools,
    get_event_tools,
    get_trading_tools,
    get_analysis_tools,
    get_read_only_tools,
    get_tool_functions,
    print_argument_reference,
)

# CLOB tools (py_clob_client library)
from agents.langchain.clob_tools import (
    get_all_clob_tools,
    get_clob_market_tools,
    get_clob_trading_tools,
    get_clob_account_tools,
    get_clob_rfq_tools,
    get_clob_readonly_tools,
    get_clob_tool_functions,
    print_clob_argument_reference,
    # Individual CLOB tools
    clob_health_check,
    clob_get_server_time,
    clob_get_midpoint,
    clob_get_price,
    clob_get_orderbook,
    clob_get_spread,
    clob_get_last_trade_price,
    clob_get_markets,
    clob_get_simplified_markets,
    clob_get_market,
    clob_create_limit_order,
    clob_create_market_order,
    clob_cancel_order,
    clob_cancel_all_orders,
    clob_get_open_orders,
    clob_get_order,
    clob_get_trades,
    clob_get_balance_allowance,
    clob_get_api_keys,
    clob_create_rfq_request,
    clob_get_rfq_requests,
    clob_create_rfq_quote,
    clob_get_rfq_quotes,
    clob_accept_rfq_quote,
    clob_cancel_rfq_request,
)


def get_combined_tools():
    """Get all tools from both agent tools and CLOB tools."""
    return get_all_tools() + get_all_clob_tools()


def get_combined_readonly_tools():
    """Get read-only tools from both agent tools and CLOB tools."""
    return get_read_only_tools() + get_clob_readonly_tools()


__all__ = [
    # Combined tool functions
    "get_combined_tools",
    "get_combined_readonly_tools",
    # Agent tool collections (polymarket-agents)
    "get_all_tools",
    "get_market_tools",
    "get_event_tools",
    "get_trading_tools",
    "get_analysis_tools",
    "get_read_only_tools",
    "get_tool_functions",
    "print_argument_reference",
    # CLOB tool collections (py_clob_client)
    "get_all_clob_tools",
    "get_clob_market_tools",
    "get_clob_trading_tools",
    "get_clob_account_tools",
    "get_clob_rfq_tools",
    "get_clob_readonly_tools",
    "get_clob_tool_functions",
    "print_clob_argument_reference",
    # Individual CLOB market tools
    "clob_health_check",
    "clob_get_server_time",
    "clob_get_midpoint",
    "clob_get_price",
    "clob_get_orderbook",
    "clob_get_spread",
    "clob_get_last_trade_price",
    "clob_get_markets",
    "clob_get_simplified_markets",
    "clob_get_market",
    # Individual CLOB trading tools
    "clob_create_limit_order",
    "clob_create_market_order",
    "clob_cancel_order",
    "clob_cancel_all_orders",
    "clob_get_open_orders",
    "clob_get_order",
    "clob_get_trades",
    # Individual CLOB account tools
    "clob_get_balance_allowance",
    "clob_get_api_keys",
    # Individual CLOB RFQ tools
    "clob_create_rfq_request",
    "clob_get_rfq_requests",
    "clob_create_rfq_quote",
    "clob_get_rfq_quotes",
    "clob_accept_rfq_quote",
    "clob_cancel_rfq_request",
]

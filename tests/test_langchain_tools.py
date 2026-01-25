"""
Tests for LangChain integration tools.

Run with:
    python -m pytest tests/test_langchain_tools.py -v
    # or
    python -m unittest tests.test_langchain_tools -v
"""

import unittest
from unittest.mock import patch, MagicMock
import json


class TestToolImports(unittest.TestCase):
    """Test that all LangChain tools can be imported."""

    def test_import_tools_module(self):
        """Test tools module imports."""
        from polymarket_agents.langchain import tools

        self.assertIsNotNone(tools)

    def test_import_get_all_tools(self):
        """Test get_all_tools function exists."""
        from polymarket_agents.langchain.tools import get_all_tools

        self.assertTrue(callable(get_all_tools))

    def test_get_all_tools_returns_list(self):
        """Test get_all_tools returns a list of tools."""
        from polymarket_agents.langchain.tools import get_all_tools

        tools = get_all_tools()
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)

    def test_get_market_tools(self):
        """Test get_market_tools function."""
        from polymarket_agents.langchain.tools import get_market_tools

        tools = get_market_tools()
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 7)

    def test_get_event_tools(self):
        """Test get_event_tools function."""
        from polymarket_agents.langchain.tools import get_event_tools

        tools = get_event_tools()
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 4)

    def test_get_read_only_tools(self):
        """Test get_read_only_tools function."""
        from polymarket_agents.langchain.tools import get_read_only_tools

        tools = get_read_only_tools()
        self.assertIsInstance(tools, list)
        self.assertGreater(len(tools), 0)

    def test_get_trading_tools(self):
        """Test get_trading_tools function."""
        from polymarket_agents.langchain.tools import get_trading_tools

        tools = get_trading_tools()
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 5)

    def test_get_analysis_tools(self):
        """Test get_analysis_tools function."""
        from polymarket_agents.langchain.tools import get_analysis_tools

        tools = get_analysis_tools()
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 7)


class TestToolSchemas(unittest.TestCase):
    """Test Pydantic schemas for tool inputs."""

    def test_market_query_input(self):
        """Test MarketQueryInput schema."""
        from polymarket_agents.langchain.tools import MarketQueryInput

        # Test with defaults
        input_default = MarketQueryInput()
        self.assertEqual(input_default.limit, 10)
        self.assertTrue(input_default.active_only)

        # Test with values
        input_custom = MarketQueryInput(limit=5, active_only=False)
        self.assertEqual(input_custom.limit, 5)
        self.assertFalse(input_custom.active_only)

    def test_order_input(self):
        """Test OrderInput schema."""
        from polymarket_agents.langchain.tools import OrderInput

        order = OrderInput(token_id="12345", price=0.65, size=100.0, side="BUY")
        self.assertEqual(order.token_id, "12345")
        self.assertEqual(order.price, 0.65)
        self.assertEqual(order.size, 100.0)
        self.assertEqual(order.side, "BUY")

    def test_forecast_input(self):
        """Test ForecastInput schema."""
        from polymarket_agents.langchain.tools import ForecastInput

        forecast = ForecastInput(
            event_title="Test Event", market_question="Will X happen?", outcome="Yes"
        )
        self.assertEqual(forecast.event_title, "Test Event")
        self.assertEqual(forecast.market_question, "Will X happen?")
        self.assertEqual(forecast.outcome, "Yes")


class TestToolFunctionality(unittest.TestCase):
    """Test individual tool functions."""

    def test_preview_order_buy(self):
        """Test preview_order for BUY side."""
        from polymarket_agents.langchain.tools import preview_order

        result = preview_order.invoke(
            {"token_id": "12345", "price": 0.65, "size": 100.0, "side": "BUY"}
        )

        data = json.loads(result)
        self.assertTrue(data["preview"])
        self.assertEqual(data["side"], "BUY")
        self.assertEqual(data["estimated_cost_usdc"], 65.0)
        self.assertEqual(data["potential_payout_usdc"], 100.0)

    def test_preview_order_sell(self):
        """Test preview_order for SELL side."""
        from polymarket_agents.langchain.tools import preview_order

        result = preview_order.invoke(
            {"token_id": "12345", "price": 0.65, "size": 100.0, "side": "SELL"}
        )

        data = json.loads(result)
        self.assertTrue(data["preview"])
        self.assertEqual(data["side"], "SELL")

    def test_preview_order_invalid_side(self):
        """Test preview_order rejects invalid side."""
        from polymarket_agents.langchain.tools import preview_order

        result = preview_order.invoke(
            {"token_id": "12345", "price": 0.65, "size": 100.0, "side": "INVALID"}
        )

        self.assertIn("Error", result)

    def test_preview_order_invalid_price(self):
        """Test preview_order rejects invalid price."""
        from polymarket_agents.langchain.tools import preview_order

        result = preview_order.invoke(
            {
                "token_id": "12345",
                "price": 1.5,  # Invalid - must be 0.01-0.99
                "size": 100.0,
                "side": "BUY",
            }
        )

        self.assertIn("Error", result)


class TestToolHasCorrectMetadata(unittest.TestCase):
    """Test that tools have proper metadata for LLM use."""

    def test_tools_have_names(self):
        """Test all tools have names."""
        from polymarket_agents.langchain.tools import get_all_tools

        for tool in get_all_tools():
            self.assertIsNotNone(tool.name)
            self.assertIsInstance(tool.name, str)
            self.assertGreater(len(tool.name), 0)

    def test_tools_have_descriptions(self):
        """Test all tools have descriptions."""
        from polymarket_agents.langchain.tools import get_all_tools

        for tool in get_all_tools():
            self.assertIsNotNone(tool.description)
            self.assertIsInstance(tool.description, str)
            self.assertGreater(len(tool.description), 10)

    def test_tool_names_are_unique(self):
        """Test all tool names are unique."""
        from polymarket_agents.langchain.tools import get_all_tools

        tools = get_all_tools()
        names = [t.name for t in tools]
        self.assertEqual(len(names), len(set(names)))


class TestAgentCreation(unittest.TestCase):
    """Test agent creation functions."""

    def test_import_agent_module(self):
        """Test agent module imports."""
        from polymarket_agents.langchain import agent

        self.assertTrue(hasattr(agent, "create_polymarket_agent"))
        self.assertTrue(hasattr(agent, "create_simple_analyst"))
        self.assertTrue(hasattr(agent, "run_agent"))

    def test_agent_functions_callable(self):
        """Test agent functions are callable."""
        from polymarket_agents.langchain.agent import (
            create_polymarket_agent,
            create_simple_analyst,
            create_research_agent,
            run_agent,
            find_best_trade,
            analyze_specific_market,
        )

        self.assertTrue(callable(create_polymarket_agent))
        self.assertTrue(callable(create_simple_analyst))
        self.assertTrue(callable(create_research_agent))
        self.assertTrue(callable(run_agent))
        self.assertTrue(callable(find_best_trade))
        self.assertTrue(callable(analyze_specific_market))


class TestArgumentReference(unittest.TestCase):
    """Test argument reference documentation."""

    def test_argument_reference_exists(self):
        """Test ARGUMENT_REFERENCE constant exists."""
        from polymarket_agents.langchain.tools import ARGUMENT_REFERENCE

        self.assertIsInstance(ARGUMENT_REFERENCE, str)
        self.assertGreater(len(ARGUMENT_REFERENCE), 100)

    def test_print_argument_reference(self):
        """Test print_argument_reference function exists."""
        from polymarket_agents.langchain.tools import print_argument_reference

        self.assertTrue(callable(print_argument_reference))


if __name__ == "__main__":
    unittest.main(verbosity=2)

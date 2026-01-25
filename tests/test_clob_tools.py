"""
Tests for CLOB LangChain Tools (py_clob_client wrapper)

These tests verify the CLOB tools module structure and imports.
Actual API calls are mocked to avoid requiring live credentials.
"""

import unittest
from unittest.mock import patch, MagicMock
import json


class TestCLOBToolImports(unittest.TestCase):
    """Test that CLOB tool imports work correctly."""

    def test_import_clob_tools_module(self):
        """Test importing the clob_tools module."""
        from polymarket_agents.langchain import clob_tools

        self.assertIsNotNone(clob_tools)

    def test_import_tool_collections(self):
        """Test importing tool collection functions."""
        from polymarket_agents.langchain.clob_tools import (
            get_all_clob_tools,
            get_clob_market_tools,
            get_clob_trading_tools,
            get_clob_account_tools,
            get_clob_rfq_tools,
            get_clob_readonly_tools,
        )

        self.assertIsNotNone(get_all_clob_tools)
        self.assertIsNotNone(get_clob_market_tools)
        self.assertIsNotNone(get_clob_trading_tools)
        self.assertIsNotNone(get_clob_account_tools)
        self.assertIsNotNone(get_clob_rfq_tools)
        self.assertIsNotNone(get_clob_readonly_tools)

    def test_import_individual_market_tools(self):
        """Test importing individual market data tools."""
        from polymarket_agents.langchain.clob_tools import (
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
        )

        self.assertIsNotNone(clob_health_check)
        self.assertIsNotNone(clob_get_server_time)
        self.assertIsNotNone(clob_get_midpoint)
        self.assertIsNotNone(clob_get_price)
        self.assertIsNotNone(clob_get_orderbook)
        self.assertIsNotNone(clob_get_spread)
        self.assertIsNotNone(clob_get_last_trade_price)
        self.assertIsNotNone(clob_get_markets)
        self.assertIsNotNone(clob_get_simplified_markets)
        self.assertIsNotNone(clob_get_market)

    def test_import_individual_trading_tools(self):
        """Test importing individual trading tools."""
        from polymarket_agents.langchain.clob_tools import (
            clob_create_limit_order,
            clob_create_market_order,
            clob_cancel_order,
            clob_cancel_all_orders,
            clob_get_open_orders,
            clob_get_order,
            clob_get_trades,
        )

        self.assertIsNotNone(clob_create_limit_order)
        self.assertIsNotNone(clob_create_market_order)
        self.assertIsNotNone(clob_cancel_order)
        self.assertIsNotNone(clob_cancel_all_orders)
        self.assertIsNotNone(clob_get_open_orders)
        self.assertIsNotNone(clob_get_order)
        self.assertIsNotNone(clob_get_trades)

    def test_import_individual_rfq_tools(self):
        """Test importing individual RFQ tools."""
        from polymarket_agents.langchain.clob_tools import (
            clob_create_rfq_request,
            clob_get_rfq_requests,
            clob_create_rfq_quote,
            clob_get_rfq_quotes,
            clob_accept_rfq_quote,
            clob_cancel_rfq_request,
        )

        self.assertIsNotNone(clob_create_rfq_request)
        self.assertIsNotNone(clob_get_rfq_requests)
        self.assertIsNotNone(clob_create_rfq_quote)
        self.assertIsNotNone(clob_get_rfq_quotes)
        self.assertIsNotNone(clob_accept_rfq_quote)
        self.assertIsNotNone(clob_cancel_rfq_request)

    def test_import_from_init(self):
        """Test importing from __init__.py."""
        from polymarket_agents.langchain import (
            get_all_clob_tools,
            get_clob_market_tools,
            get_combined_tools,
            get_combined_readonly_tools,
        )

        self.assertIsNotNone(get_all_clob_tools)
        self.assertIsNotNone(get_clob_market_tools)
        self.assertIsNotNone(get_combined_tools)
        self.assertIsNotNone(get_combined_readonly_tools)


class TestCLOBToolCollections(unittest.TestCase):
    """Test tool collection functions."""

    def test_get_clob_market_tools_count(self):
        """Test market tools collection has expected count."""
        from polymarket_agents.langchain.clob_tools import get_clob_market_tools

        tools = get_clob_market_tools()
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 10)

    def test_get_clob_trading_tools_count(self):
        """Test trading tools collection has expected count."""
        from polymarket_agents.langchain.clob_tools import get_clob_trading_tools

        tools = get_clob_trading_tools()
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 7)

    def test_get_clob_account_tools_count(self):
        """Test account tools collection has expected count."""
        from polymarket_agents.langchain.clob_tools import get_clob_account_tools

        tools = get_clob_account_tools()
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 2)

    def test_get_clob_rfq_tools_count(self):
        """Test RFQ tools collection has expected count."""
        from polymarket_agents.langchain.clob_tools import get_clob_rfq_tools

        tools = get_clob_rfq_tools()
        self.assertIsInstance(tools, list)
        self.assertEqual(len(tools), 6)

    def test_get_all_clob_tools_count(self):
        """Test all CLOB tools collection has expected total."""
        from polymarket_agents.langchain.clob_tools import get_all_clob_tools

        tools = get_all_clob_tools()
        self.assertIsInstance(tools, list)
        # 10 market + 7 trading + 2 account + 6 rfq = 25
        self.assertEqual(len(tools), 25)

    def test_get_clob_readonly_tools(self):
        """Test readonly tools only includes market tools."""
        from polymarket_agents.langchain.clob_tools import (
            get_clob_readonly_tools,
            get_clob_market_tools,
        )

        readonly = get_clob_readonly_tools()
        market = get_clob_market_tools()
        self.assertEqual(len(readonly), len(market))


class TestCLOBToolsHaveRequiredAttributes(unittest.TestCase):
    """Test that tools have required LangChain attributes."""

    def test_market_tools_have_name_and_description(self):
        """Test market tools have name and description."""
        from polymarket_agents.langchain.clob_tools import get_clob_market_tools

        tools = get_clob_market_tools()
        for tool in tools:
            self.assertTrue(hasattr(tool, "name"), f"{tool} missing name")
            self.assertTrue(hasattr(tool, "description"), f"{tool} missing description")
            self.assertTrue(
                len(tool.description) > 0, f"{tool.name} has empty description"
            )

    def test_trading_tools_have_name_and_description(self):
        """Test trading tools have name and description."""
        from polymarket_agents.langchain.clob_tools import get_clob_trading_tools

        tools = get_clob_trading_tools()
        for tool in tools:
            self.assertTrue(hasattr(tool, "name"), f"{tool} missing name")
            self.assertTrue(hasattr(tool, "description"), f"{tool} missing description")

    def test_rfq_tools_have_name_and_description(self):
        """Test RFQ tools have name and description."""
        from polymarket_agents.langchain.clob_tools import get_clob_rfq_tools

        tools = get_clob_rfq_tools()
        for tool in tools:
            self.assertTrue(hasattr(tool, "name"), f"{tool} missing name")
            self.assertTrue(hasattr(tool, "description"), f"{tool} missing description")


class TestCLOBToolArgumentReference(unittest.TestCase):
    """Test argument reference documentation."""

    def test_argument_reference_exists(self):
        """Test CLOB argument reference string exists."""
        from polymarket_agents.langchain.clob_tools import CLOB_ARGUMENT_REFERENCE

        self.assertIsNotNone(CLOB_ARGUMENT_REFERENCE)
        self.assertIn("CLOB TOOLS", CLOB_ARGUMENT_REFERENCE)
        self.assertIn("MARKET DATA TOOLS", CLOB_ARGUMENT_REFERENCE)
        self.assertIn("TRADING TOOLS", CLOB_ARGUMENT_REFERENCE)
        self.assertIn("RFQ TOOLS", CLOB_ARGUMENT_REFERENCE)

    def test_print_argument_reference_callable(self):
        """Test print_clob_argument_reference is callable."""
        from polymarket_agents.langchain.clob_tools import print_clob_argument_reference

        self.assertTrue(callable(print_clob_argument_reference))


class TestCLOBCombinedTools(unittest.TestCase):
    """Test combined tool functions."""

    def test_get_combined_tools(self):
        """Test get_combined_tools returns both agent and CLOB tools."""
        from polymarket_agents.langchain import get_combined_tools, get_all_tools
        from polymarket_agents.langchain.clob_tools import get_all_clob_tools

        combined = get_combined_tools()
        agent_tools = get_all_tools()
        clob_tools = get_all_clob_tools()

        self.assertEqual(len(combined), len(agent_tools) + len(clob_tools))

    def test_get_combined_readonly_tools(self):
        """Test get_combined_readonly_tools returns readonly tools from both."""
        from polymarket_agents.langchain import get_combined_readonly_tools, get_read_only_tools
        from polymarket_agents.langchain.clob_tools import get_clob_readonly_tools

        combined = get_combined_readonly_tools()
        agent_readonly = get_read_only_tools()
        clob_readonly = get_clob_readonly_tools()

        self.assertEqual(len(combined), len(agent_readonly) + len(clob_readonly))


class TestCLOBToolsMocked(unittest.TestCase):
    """Test CLOB tools with mocked client."""

    @patch("polymarket_agents.langchain.clob_tools._clob_client_readonly")
    def test_health_check_with_mock(self, mock_client):
        """Test health check tool with mocked client."""
        from polymarket_agents.langchain.clob_tools import clob_health_check
        import polymarket_agents.langchain.clob_tools as clob_module

        # Reset the global client
        clob_module._clob_client_readonly = None

        mock_instance = MagicMock()
        mock_instance.get_ok.return_value = "OK"

        with patch.object(
            clob_module, "_get_clob_client_readonly", return_value=mock_instance
        ):
            result = clob_health_check.invoke({})
            self.assertIn("OK", result)

    @patch("polymarket_agents.langchain.clob_tools._clob_client_readonly")
    def test_get_midpoint_with_mock(self, mock_client):
        """Test get_midpoint tool with mocked client."""
        from polymarket_agents.langchain.clob_tools import clob_get_midpoint
        import polymarket_agents.langchain.clob_tools as clob_module

        clob_module._clob_client_readonly = None

        mock_instance = MagicMock()
        mock_instance.get_midpoint.return_value = {"mid": "0.55"}

        with patch.object(
            clob_module, "_get_clob_client_readonly", return_value=mock_instance
        ):
            result = clob_get_midpoint.invoke({"token_id": "12345"})
            self.assertIn("0.55", result)


class TestCLOBPydanticSchemas(unittest.TestCase):
    """Test Pydantic schema definitions."""

    def test_clob_order_args_schema(self):
        """Test CLOBOrderArgs schema."""
        from polymarket_agents.langchain.clob_tools import CLOBOrderArgs

        # Valid order
        order = CLOBOrderArgs(token_id="12345", price=0.50, size=10.0, side="BUY")
        self.assertEqual(order.token_id, "12345")
        self.assertEqual(order.price, 0.50)
        self.assertEqual(order.size, 10.0)
        self.assertEqual(order.side, "BUY")

    def test_clob_market_order_args_schema(self):
        """Test CLOBMarketOrderArgs schema."""
        from polymarket_agents.langchain.clob_tools import CLOBMarketOrderArgs

        order = CLOBMarketOrderArgs(token_id="12345", amount=100.0, side="SELL")
        self.assertEqual(order.token_id, "12345")
        self.assertEqual(order.amount, 100.0)
        self.assertEqual(order.side, "SELL")


if __name__ == "__main__":
    unittest.main()

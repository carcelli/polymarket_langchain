import sys
import os
import unittest
from unittest.mock import MagicMock, patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.tools.trade_tools import (
    execute_market_order,
    execute_limit_order,
    cancel_all_orders,
)
from py_clob_client.order_builder.constants import SELL


class TestTradeTools(unittest.TestCase):

    @patch("agents.tools.trade_tools.Polymarket")
    def test_execute_market_buy_mock(self, MockPolymarket):
        # Setup mock for BUY
        mock_poly = MockPolymarket.return_value
        mock_poly.client = MagicMock()
        mock_poly.client.create_market_order.return_value = "signed_buy_order"
        mock_poly.client.post_order.return_value = {
            "status": "success",
            "orderID": "buy_1",
        }

        # Test BUY invocation
        result = execute_market_order.invoke(
            {"token_id": "12345", "amount": 10.0, "side": "BUY"}
        )

        print(f"Market Buy Result: {result}")
        self.assertIn("success", result)
        self.assertIn("buy_1", result)
        # Verify create_market_order was called (implicit check that it didn't fail on args)
        mock_poly.client.create_market_order.assert_called_once()

    @patch("agents.tools.trade_tools.Polymarket")
    def test_execute_market_sell_mock(self, MockPolymarket):
        # Setup mock for SELL
        mock_poly = MockPolymarket.return_value
        mock_poly.client = MagicMock()
        mock_poly.client.create_order.return_value = "signed_sell_order"
        mock_poly.client.post_order.return_value = {
            "status": "success",
            "orderID": "sell_1",
        }

        # Test SELL invocation
        result = execute_market_order.invoke(
            {"token_id": "12345", "amount": 5.0, "side": "SELL"}  # shares
        )

        print(f"Market Sell Result: {result}")
        self.assertIn("success", result)
        self.assertIn("sell_1", result)

        # Verify create_order was called (Limit order logic)
        mock_poly.client.create_order.assert_called_once()
        args, _ = mock_poly.client.create_order.call_args
        order_args = args[0]
        self.assertEqual(order_args.side, SELL)
        self.assertEqual(order_args.size, 5.0)
        self.assertEqual(order_args.price, 0.01)

    @patch("agents.tools.trade_tools.Polymarket")
    def test_cancel_all_mock(self, MockPolymarket):
        mock_poly = MockPolymarket.return_value
        mock_poly.client = MagicMock()
        mock_poly.client.cancel_all.return_value = {"cancelled": True}

        result = cancel_all_orders.invoke({})
        print(f"Cancel All Result: {result}")
        self.assertIn("cancelled", result)


if __name__ == "__main__":
    unittest.main()

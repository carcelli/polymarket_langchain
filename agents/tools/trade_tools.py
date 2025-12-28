import json
from typing import Optional

from agents.tooling import wrap_tool
from agents.polymarket.polymarket import Polymarket
from py_clob_client.clob_types import OrderArgs, MarketOrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY, SELL

# Singleton instance to reuse connection
_poly_client: Optional[Polymarket] = None
_poly_client_factory: Optional[object] = None

def get_poly_client() -> Polymarket:
    global _poly_client, _poly_client_factory
    if _poly_client is None or _poly_client_factory is not Polymarket:
        try:
            _poly_client = Polymarket()
            _poly_client_factory = Polymarket
        except Exception as e:
            # We fail gracefully if creds aren't set, tools will report error on use
            print(f"Warning: Failed to initialize Polymarket client: {e}")
            _poly_client = None
    return _poly_client

def _execute_market_order_impl(token_id: str, amount: float, side: str) -> str:
    """
    Executes a MARKET order (FOK).
    
    Args:
        token_id: The CLOB Token ID of the outcome to trade.
        amount: For BUY orders, this is the amount of USDC to spend.
                For SELL orders, this is the number of SHARES to sell.
        side: 'BUY' or 'SELL'
    
    Returns:
        JSON string with the order response or error.
    """
    try:
        poly = get_poly_client()
        if not poly or not poly.client:
            return json.dumps({"error": "Polymarket client not initialized (check private key)"})

        if side.upper() == "BUY":
            # Market Buy (Spend USDC)
            # Installed py_clob_client MarketOrderArgs seems to lack 'side' arg, implying Buy/Spend-USDC default
            order_args = MarketOrderArgs(
                token_id=token_id,
                amount=amount
            )
            signed_order = poly.client.create_market_order(order_args)
            resp = poly.client.post_order(signed_order, orderType=OrderType.FOK)
            
        else: # SELL
            # Market Sell (Sell Shares)
            # Implemented as Limit Sell at min price (0.01) with FOK
            order_args = OrderArgs(
                price=0.01,
                size=amount,
                side=SELL,
                token_id=token_id
            )
            # We use create_order for Limit order construction
            signed_order = poly.client.create_order(order_args)
            resp = poly.client.post_order(signed_order, orderType=OrderType.FOK)
            
        return json.dumps(resp, default=str)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to execute market order: {str(e)}"})

def _execute_limit_order_impl(token_id: str, price: float, size: float, side: str) -> str:
    """
    Executes a LIMIT order (GTC - Good Til Cancelled).
    
    Args:
        token_id: The CLOB Token ID of the outcome.
        price: The limit price (0.0 to 1.0).
        size: The number of shares to buy/sell.
        side: 'BUY' or 'SELL'.
    
    Returns:
        JSON string with the order response or error.
    """
    try:
        poly = get_poly_client()
        if not poly or not poly.client:
            return json.dumps({"error": "Polymarket client not initialized"})

        side_const = BUY if side.upper() == "BUY" else SELL
        
        # Poly's client.create_and_post_order is a helper for this
        resp = poly.client.create_and_post_order(
            OrderArgs(
                price=price,
                size=size,
                side=side_const,
                token_id=token_id
            )
        )
        return json.dumps(resp, default=str)
        
    except Exception as e:
        return json.dumps({"error": f"Failed to execute limit order: {str(e)}"})

def _cancel_all_orders_impl() -> str:
    """
    Cancels ALL open orders. Use this for emergency stops or resetting state.
    """
    try:
        poly = get_poly_client()
        if not poly or not poly.client:
            return json.dumps({"error": "Polymarket client not initialized"})
            
        resp = poly.client.cancel_all()
        return json.dumps(resp, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to cancel all orders: {str(e)}"})

def _cancel_order_impl(order_id: str) -> str:
    """
    Cancels a specific order by its ID.
    """
    try:
        poly = get_poly_client()
        if not poly or not poly.client:
            return json.dumps({"error": "Polymarket client not initialized"})
            
        resp = poly.client.cancel(order_id)
        return json.dumps(resp, default=str)
    except Exception as e:
        return json.dumps({"error": f"Failed to cancel order {order_id}: {str(e)}"})


execute_market_order = wrap_tool(
    _execute_market_order_impl,
    name="execute_market_order",
)
execute_limit_order = wrap_tool(
    _execute_limit_order_impl,
    name="execute_limit_order",
)
cancel_all_orders = wrap_tool(
    _cancel_all_orders_impl,
    name="cancel_all_orders",
)
cancel_order = wrap_tool(
    _cancel_order_impl,
    name="cancel_order",
)

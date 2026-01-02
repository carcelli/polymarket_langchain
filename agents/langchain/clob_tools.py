"""
LangChain Tools for py_clob_client (Official Polymarket CLOB Client)

This module wraps the official Polymarket py_clob_client library as LangChain tools.
These tools provide direct access to the Central Limit Order Book (CLOB) API.

Tool implementations are defined as plain functions and wrapped via wrap_tool
to keep compatibility across different runtime configurations.

AUTHENTICATION LEVELS:
- Level 0: No auth - market data only (prices, orderbooks)
- Level 1: Private key - can derive API keys
- Level 2: Full API creds - can trade, manage orders

ENVIRONMENT VARIABLES:
- CLOB_API_URL: CLOB API endpoint (default: https://clob.polymarket.com)
- PK or POLYGON_WALLET_PRIVATE_KEY: Wallet private key
- CLOB_API_KEY: API key (or derive from private key)
- CLOB_SECRET: API secret
- CLOB_PASS_PHRASE: API passphrase

Usage:
    from agents.langchain.clob_tools import get_clob_market_tools, get_clob_trading_tools

    # Read-only tools (no auth required)
    tools = get_clob_market_tools()

    # Trading tools (requires full auth)
    tools = get_clob_trading_tools()
"""

import json
import os
from typing import List, Dict, Callable

from pydantic import BaseModel, Field

from agents.tooling import wrap_tool

# =============================================================================
# LAZY CLIENT INITIALIZATION
# =============================================================================

_clob_client = None
_clob_client_readonly = None


def _get_clob_client_readonly():
    """Get read-only CLOB client (no auth required)."""
    global _clob_client_readonly
    if _clob_client_readonly is None:
        try:
            from py_clob_client.client import ClobClient

            host = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
            _clob_client_readonly = ClobClient(host)
        except ImportError:
            raise ImportError(
                "py_clob_client not installed. Install with: pip install py-clob-client"
            )
    return _clob_client_readonly


def _get_clob_client():
    """Get authenticated CLOB client (requires API credentials)."""
    global _clob_client
    if _clob_client is None:
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
            from py_clob_client.constants import POLYGON

            host = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
            key = os.getenv("PK") or os.getenv("POLYGON_WALLET_PRIVATE_KEY")
            chain_id = int(os.getenv("CHAIN_ID", POLYGON))

            if not key:
                raise ValueError(
                    "Private key required. Set PK or POLYGON_WALLET_PRIVATE_KEY env var."
                )

            # Check for existing API creds or derive them
            api_key = os.getenv("CLOB_API_KEY")
            api_secret = os.getenv("CLOB_SECRET")
            api_passphrase = os.getenv("CLOB_PASS_PHRASE")

            if api_key and api_secret and api_passphrase:
                creds = ApiCreds(
                    api_key=api_key,
                    api_secret=api_secret,
                    api_passphrase=api_passphrase,
                )
                _clob_client = ClobClient(host, key=key, chain_id=chain_id, creds=creds)
            else:
                # Create client and derive credentials
                _clob_client = ClobClient(host, key=key, chain_id=chain_id)
                creds = _clob_client.create_or_derive_api_creds()
                _clob_client.set_api_creds(creds)

        except ImportError:
            raise ImportError(
                "py_clob_client not installed. Install with: pip install py-clob-client"
            )
    return _clob_client


# =============================================================================
# PYDANTIC SCHEMAS
# =============================================================================


class CLOBOrderArgs(BaseModel):
    """Schema for CLOB limit order arguments."""

    token_id: str = Field(description="Token ID of the conditional token")
    price: float = Field(ge=0.01, le=0.99, description="Price per share (0.01 to 0.99)")
    size: float = Field(gt=0, description="Number of shares to trade")
    side: str = Field(pattern="^(BUY|SELL)$", description="BUY or SELL")


class CLOBMarketOrderArgs(BaseModel):
    """Schema for CLOB market order arguments."""

    token_id: str = Field(description="Token ID of the conditional token")
    amount: float = Field(gt=0, description="USDC amount (BUY) or shares (SELL)")
    side: str = Field(pattern="^(BUY|SELL)$", description="BUY or SELL")


# =============================================================================
# CLOB MARKET DATA TOOLS (READ-ONLY, NO AUTH)
# =============================================================================


def _clob_health_check_impl() -> str:
    """Check if the Polymarket CLOB API is healthy and responding.

    Use this to verify connectivity before making other API calls.

    Returns:
        "OK" if the API is healthy, error message otherwise
    """
    try:
        client = _get_clob_client_readonly()
        result = client.get_ok()
        return f"CLOB API Status: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def _clob_get_server_time_impl() -> str:
    """Get the current server time from the CLOB API.

    Useful for debugging timing issues or verifying API connectivity.

    Returns:
        Server timestamp
    """
    try:
        client = _get_clob_client_readonly()
        result = client.get_server_time()
        return f"Server Time: {result}"
    except Exception as e:
        return f"Error: {str(e)}"


def _clob_get_midpoint_impl(token_id: str) -> str:
    """Get the mid-market price for a token.

    The midpoint is the average of best bid and best ask prices.
    Use this for a quick price check without full orderbook data.

    Args:
        token_id: The conditional token ID (long numeric string)

    Returns:
        Mid-market price as a decimal (e.g., "0.55" = 55% probability)
    """
    try:
        client = _get_clob_client_readonly()
        result = client.get_midpoint(token_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def _clob_get_price_impl(token_id: str, side: str) -> str:
    """Get the current price for buying or selling a token.

    Returns the best available price for the specified side.
    BUY price is the lowest ask, SELL price is the highest bid.

    Args:
        token_id: The conditional token ID
        side: "BUY" or "SELL"

    Returns:
        Current price for the specified side
    """
    try:
        client = _get_clob_client_readonly()
        side = side.upper()
        if side not in ["BUY", "SELL"]:
            return "Error: side must be 'BUY' or 'SELL'"
        result = client.get_price(token_id, side)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def _clob_get_orderbook_impl(token_id: str) -> str:
    """Get the full order book for a token.

    Shows all open bids (buy orders) and asks (sell orders) with
    their prices and sizes. Essential for understanding market depth.

    Args:
        token_id: The conditional token ID

    Returns:
        JSON with bids and asks arrays, each containing price/size
    """
    try:
        client = _get_clob_client_readonly()
        orderbook = client.get_order_book(token_id)
        return json.dumps(
            {
                "market": orderbook.market,
                "asset_id": orderbook.asset_id,
                "bids": [
                    {"price": b.price, "size": b.size} for b in orderbook.bids[:15]
                ],
                "asks": [
                    {"price": a.price, "size": a.size} for a in orderbook.asks[:15]
                ],
                "hash": orderbook.hash,
            },
            indent=2,
        )
    except Exception as e:
        return f"Error: {str(e)}"


def _clob_get_spread_impl(token_id: str) -> str:
    """Get the bid-ask spread for a token.

    The spread is the difference between the best ask and best bid.
    Tighter spreads indicate more liquid markets.

    Args:
        token_id: The conditional token ID

    Returns:
        Spread information with best bid, best ask, and spread size
    """
    try:
        client = _get_clob_client_readonly()
        result = client.get_spread(token_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def _clob_get_last_trade_price_impl(token_id: str) -> str:
    """Get the last trade price for a token.

    Shows the price at which the most recent trade occurred.

    Args:
        token_id: The conditional token ID

    Returns:
        Last trade price
    """
    try:
        client = _get_clob_client_readonly()
        result = client.get_last_trade_price(token_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def _clob_get_markets_impl(next_cursor: str = "") -> str:
    """Get all markets from the CLOB API.

    Returns a paginated list of all available markets.
    Use the next_cursor for pagination through results.

    Args:
        next_cursor: Pagination cursor (empty for first page)

    Returns:
        JSON with markets data and pagination cursor
    """
    try:
        client = _get_clob_client_readonly()
        result = client.get_markets(next_cursor=next_cursor if next_cursor else None)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def _clob_get_simplified_markets_impl(next_cursor: str = "") -> str:
    """Get simplified market data from the CLOB API.

    Returns a lighter-weight response with essential market info.
    Better for quick lookups when you don't need full details.

    Args:
        next_cursor: Pagination cursor (empty for first page)

    Returns:
        JSON with simplified markets data
    """
    try:
        client = _get_clob_client_readonly()
        result = client.get_simplified_markets(
            next_cursor=next_cursor if next_cursor else None
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


def _clob_get_market_impl(condition_id: str) -> str:
    """Get detailed information about a specific market.

    Args:
        condition_id: The market condition ID

    Returns:
        Full market details including tokens, outcomes, etc.
    """
    try:
        client = _get_clob_client_readonly()
        result = client.get_market(condition_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# CLOB TRADING TOOLS (REQUIRE AUTHENTICATION)
# =============================================================================


def _clob_create_limit_order_impl(
    token_id: str, price: float, size: float, side: str
) -> str:
    """Create and submit a limit order on Polymarket CLOB.

    WARNING: This executes a REAL trade with REAL money!

    A limit order specifies the exact price you want to trade at.
    The order will only execute if the market reaches your price.

    Args:
        token_id: The conditional token ID to trade
        price: Price per share (0.01 to 0.99)
        size: Number of shares to trade
        side: "BUY" or "SELL"

    Returns:
        Order confirmation with order ID, or error message
    """
    try:
        from py_clob_client.clob_types import OrderArgs, OrderType

        client = _get_clob_client()
        side = side.upper()

        if side not in ["BUY", "SELL"]:
            return "Error: side must be 'BUY' or 'SELL'"
        if not 0.01 <= price <= 0.99:
            return "Error: price must be between 0.01 and 0.99"
        if size <= 0:
            return "Error: size must be positive"

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
        )

        signed_order = client.create_order(order_args)
        result = client.post_order(signed_order, OrderType.GTC)

        return json.dumps(
            {
                "success": True,
                "order_type": "LIMIT",
                "side": side,
                "price": price,
                "size": size,
                "token_id": token_id,
                "response": result,
            },
            indent=2,
        )

    except Exception as e:
        return f"Error creating limit order: {str(e)}"


def _clob_create_market_order_impl(token_id: str, amount: float, side: str) -> str:
    """Create and submit a market order on Polymarket CLOB.

    WARNING: This executes a REAL trade with REAL money!

    A market order executes immediately at the best available price.
    - BUY: amount is in USDC (how much $ to spend)
    - SELL: amount is in shares (how many shares to sell)

    Args:
        token_id: The conditional token ID to trade
        amount: USDC amount (BUY) or number of shares (SELL)
        side: "BUY" or "SELL"

    Returns:
        Order confirmation or error message
    """
    try:
        from py_clob_client.clob_types import MarketOrderArgs, OrderType

        client = _get_clob_client()
        side = side.upper()

        if side not in ["BUY", "SELL"]:
            return "Error: side must be 'BUY' or 'SELL'"
        if amount <= 0:
            return "Error: amount must be positive"

        order_args = MarketOrderArgs(
            token_id=token_id,
            amount=amount,
            side=side,
        )

        signed_order = client.create_market_order(order_args)
        result = client.post_order(signed_order, OrderType.FOK)

        return json.dumps(
            {
                "success": True,
                "order_type": "MARKET",
                "side": side,
                "amount": amount,
                "token_id": token_id,
                "response": result,
            },
            indent=2,
        )

    except Exception as e:
        return f"Error creating market order: {str(e)}"


def _clob_cancel_order_impl(order_id: str) -> str:
    """Cancel a specific open order.

    Args:
        order_id: The order ID to cancel (0x... hash)

    Returns:
        Cancellation confirmation or error message
    """
    try:
        client = _get_clob_client()
        result = client.cancel(order_id=order_id)
        return json.dumps(
            {
                "success": True,
                "cancelled_order": order_id,
                "response": result,
            },
            indent=2,
        )
    except Exception as e:
        return f"Error cancelling order: {str(e)}"


def _clob_cancel_all_orders_impl() -> str:
    """Cancel all open orders for the authenticated user.

    Use with caution - this cancels ALL your open orders across all markets.

    Returns:
        Cancellation confirmation or error message
    """
    try:
        client = _get_clob_client()
        result = client.cancel_all()
        return json.dumps(
            {
                "success": True,
                "message": "All orders cancelled",
                "response": result,
            },
            indent=2,
        )
    except Exception as e:
        return f"Error cancelling all orders: {str(e)}"


def _clob_get_open_orders_impl(market: str = "", asset_id: str = "") -> str:
    """Get all open orders for the authenticated user.

    Can optionally filter by market or asset.

    Args:
        market: Optional market condition ID to filter
        asset_id: Optional asset/token ID to filter

    Returns:
        JSON array of open orders with details
    """
    try:
        from py_clob_client.clob_types import OpenOrderParams

        client = _get_clob_client()

        params = OpenOrderParams()
        if market:
            params.market = market
        if asset_id:
            params.asset_id = asset_id

        result = client.get_orders(params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching open orders: {str(e)}"


def _clob_get_order_impl(order_id: str) -> str:
    """Get details of a specific order.

    Args:
        order_id: The order ID (0x... hash)

    Returns:
        Order details including status, fill amount, etc.
    """
    try:
        client = _get_clob_client()
        result = client.get_order(order_id)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching order: {str(e)}"


def _clob_get_trades_impl(market: str = "", maker_address: str = "") -> str:
    """Get trade history for the authenticated user.

    Shows completed trades with execution details.

    Args:
        market: Optional market condition ID to filter
        maker_address: Optional maker address to filter

    Returns:
        JSON array of trades
    """
    try:
        from py_clob_client.clob_types import TradeParams

        client = _get_clob_client()

        params = TradeParams()
        if market:
            params.market = market
        if maker_address:
            params.maker_address = maker_address
        else:
            # Default to current user's address
            params.maker_address = client.get_address()

        result = client.get_trades(params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching trades: {str(e)}"


# =============================================================================
# CLOB ACCOUNT TOOLS
# =============================================================================


def _clob_get_balance_allowance_impl(
    asset_type: str = "COLLATERAL",
    token_id: str = "",
) -> str:
    """Get balance and allowance for an asset.

    Checks how much of an asset you have and how much the exchange
    is allowed to spend on your behalf.

    Args:
        asset_type: "COLLATERAL" (USDC) or "CONDITIONAL" (outcome tokens)
        token_id: Required if asset_type is "CONDITIONAL"

    Returns:
        Balance and allowance information
    """
    try:
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

        client = _get_clob_client()

        if asset_type.upper() == "COLLATERAL":
            params = BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        elif asset_type.upper() == "CONDITIONAL":
            if not token_id:
                return "Error: token_id required for CONDITIONAL asset type"
            params = BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL, token_id=token_id
            )
        else:
            return "Error: asset_type must be 'COLLATERAL' or 'CONDITIONAL'"

        result = client.get_balance_allowance(params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching balance: {str(e)}"


def _clob_get_api_keys_impl() -> str:
    """Get all API keys for the authenticated user.

    Returns:
        List of API keys with their permissions
    """
    try:
        client = _get_clob_client()
        result = client.get_api_keys()
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching API keys: {str(e)}"


# =============================================================================
# CLOB RFQ (REQUEST FOR QUOTE) TOOLS
# =============================================================================


def _clob_create_rfq_request_impl(
    token_id: str, price: float, size: float, side: str
) -> str:
    """Create an RFQ (Request for Quote) request.

    RFQ allows you to request quotes from market makers for large orders
    that might move the market if executed directly on the orderbook.

    Args:
        token_id: The conditional token ID
        price: Desired price per share (0.01 to 0.99)
        size: Number of shares
        side: "BUY" or "SELL"

    Returns:
        RFQ request ID and details
    """
    try:
        from py_clob_client.rfq import RfqUserRequest

        client = _get_clob_client()
        side = side.upper()

        if side not in ["BUY", "SELL"]:
            return "Error: side must be 'BUY' or 'SELL'"

        user_request = RfqUserRequest(
            token_id=token_id,
            price=price,
            side=side,
            size=size,
        )

        result = client.rfq.create_rfq_request(user_request)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error creating RFQ request: {str(e)}"


def _clob_get_rfq_requests_impl(state: str = "active", limit: int = 10) -> str:
    """Get RFQ requests.

    Args:
        state: Filter by state ("active" or "inactive")
        limit: Maximum number of requests to return

    Returns:
        JSON array of RFQ requests
    """
    try:
        from py_clob_client.rfq import GetRfqRequestsParams

        client = _get_clob_client()

        params = GetRfqRequestsParams(
            state=state,
            limit=limit,
        )

        result = client.rfq.get_rfq_requests(params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching RFQ requests: {str(e)}"


def _clob_create_rfq_quote_impl(
    request_id: str, token_id: str, price: float, size: float, side: str
) -> str:
    """Create a quote in response to an RFQ request.

    Market makers use this to respond to RFQ requests with their prices.

    Args:
        request_id: The RFQ request ID to quote
        token_id: The conditional token ID
        price: Quoted price per share
        size: Number of shares
        side: "BUY" or "SELL" (quoter's side)

    Returns:
        Quote ID and details
    """
    try:
        from py_clob_client.rfq import RfqUserQuote

        client = _get_clob_client()
        side = side.upper()

        if side not in ["BUY", "SELL"]:
            return "Error: side must be 'BUY' or 'SELL'"

        user_quote = RfqUserQuote(
            request_id=request_id,
            token_id=token_id,
            price=price,
            side=side,
            size=size,
        )

        result = client.rfq.create_rfq_quote(user_quote)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error creating RFQ quote: {str(e)}"


def _clob_get_rfq_quotes_impl(
    request_id: str = "",
    state: str = "",
    limit: int = 10,
) -> str:
    """Get RFQ quotes.

    Args:
        request_id: Optional filter by request ID
        state: Optional filter by state
        limit: Maximum number of quotes to return

    Returns:
        JSON array of RFQ quotes
    """
    try:
        from py_clob_client.rfq import GetRfqQuotesParams

        client = _get_clob_client()

        params = GetRfqQuotesParams(limit=limit)
        if request_id:
            params.request_ids = [request_id]
        if state:
            params.state = state

        result = client.rfq.get_rfq_quotes(params)
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error fetching RFQ quotes: {str(e)}"


def _clob_accept_rfq_quote_impl(
    request_id: str, quote_id: str, expiration_seconds: int = 3600
) -> str:
    """Accept an RFQ quote (requester side).

    When you receive quotes for your RFQ request, use this to accept one.

    Args:
        request_id: The RFQ request ID
        quote_id: The quote ID to accept
        expiration_seconds: Order expiration in seconds (default: 1 hour)

    Returns:
        Acceptance confirmation
    """
    try:
        from py_clob_client.rfq import AcceptQuoteParams
        import time

        client = _get_clob_client()

        expiration = int(time.time()) + expiration_seconds

        params = AcceptQuoteParams(
            request_id=request_id,
            quote_id=quote_id,
            expiration=expiration,
        )

        result = client.rfq.accept_rfq_quote(params)
        return json.dumps(
            {
                "success": True,
                "request_id": request_id,
                "quote_id": quote_id,
                "response": result,
            },
            indent=2,
        )
    except Exception as e:
        return f"Error accepting RFQ quote: {str(e)}"


def _clob_cancel_rfq_request_impl(request_id: str) -> str:
    """Cancel an RFQ request.

    Args:
        request_id: The RFQ request ID to cancel

    Returns:
        Cancellation confirmation
    """
    try:
        from py_clob_client.rfq import CancelRfqRequestParams

        client = _get_clob_client()

        params = CancelRfqRequestParams(request_id=request_id)
        result = client.rfq.cancel_rfq_request(params)

        return json.dumps(
            {
                "success": True,
                "cancelled_request": request_id,
                "response": result,
            },
            indent=2,
        )
    except Exception as e:
        return f"Error cancelling RFQ request: {str(e)}"


# =============================================================================
# TOOL WRAPPERS
# =============================================================================

clob_health_check = wrap_tool(_clob_health_check_impl, name="clob_health_check")
clob_get_server_time = wrap_tool(
    _clob_get_server_time_impl,
    name="clob_get_server_time",
)
clob_get_midpoint = wrap_tool(_clob_get_midpoint_impl, name="clob_get_midpoint")
clob_get_price = wrap_tool(_clob_get_price_impl, name="clob_get_price")
clob_get_orderbook = wrap_tool(_clob_get_orderbook_impl, name="clob_get_orderbook")
clob_get_spread = wrap_tool(_clob_get_spread_impl, name="clob_get_spread")
clob_get_last_trade_price = wrap_tool(
    _clob_get_last_trade_price_impl,
    name="clob_get_last_trade_price",
)
clob_get_markets = wrap_tool(_clob_get_markets_impl, name="clob_get_markets")
clob_get_simplified_markets = wrap_tool(
    _clob_get_simplified_markets_impl,
    name="clob_get_simplified_markets",
)
clob_get_market = wrap_tool(_clob_get_market_impl, name="clob_get_market")

clob_create_limit_order = wrap_tool(
    _clob_create_limit_order_impl,
    name="clob_create_limit_order",
    args_schema=CLOBOrderArgs,
)
clob_create_market_order = wrap_tool(
    _clob_create_market_order_impl,
    name="clob_create_market_order",
    args_schema=CLOBMarketOrderArgs,
)
clob_cancel_order = wrap_tool(_clob_cancel_order_impl, name="clob_cancel_order")
clob_cancel_all_orders = wrap_tool(
    _clob_cancel_all_orders_impl,
    name="clob_cancel_all_orders",
)
clob_get_open_orders = wrap_tool(
    _clob_get_open_orders_impl,
    name="clob_get_open_orders",
)
clob_get_order = wrap_tool(_clob_get_order_impl, name="clob_get_order")
clob_get_trades = wrap_tool(_clob_get_trades_impl, name="clob_get_trades")
clob_get_balance_allowance = wrap_tool(
    _clob_get_balance_allowance_impl,
    name="clob_get_balance_allowance",
)
clob_get_api_keys = wrap_tool(_clob_get_api_keys_impl, name="clob_get_api_keys")

clob_create_rfq_request = wrap_tool(
    _clob_create_rfq_request_impl,
    name="clob_create_rfq_request",
)
clob_get_rfq_requests = wrap_tool(
    _clob_get_rfq_requests_impl,
    name="clob_get_rfq_requests",
)
clob_create_rfq_quote = wrap_tool(
    _clob_create_rfq_quote_impl,
    name="clob_create_rfq_quote",
)
clob_get_rfq_quotes = wrap_tool(
    _clob_get_rfq_quotes_impl,
    name="clob_get_rfq_quotes",
)
clob_accept_rfq_quote = wrap_tool(
    _clob_accept_rfq_quote_impl,
    name="clob_accept_rfq_quote",
)
clob_cancel_rfq_request = wrap_tool(
    _clob_cancel_rfq_request_impl,
    name="clob_cancel_rfq_request",
)

_CLOB_TOOL_FUNCTIONS: Dict[str, Callable] = {
    "clob_health_check": _clob_health_check_impl,
    "clob_get_server_time": _clob_get_server_time_impl,
    "clob_get_midpoint": _clob_get_midpoint_impl,
    "clob_get_price": _clob_get_price_impl,
    "clob_get_orderbook": _clob_get_orderbook_impl,
    "clob_get_spread": _clob_get_spread_impl,
    "clob_get_last_trade_price": _clob_get_last_trade_price_impl,
    "clob_get_markets": _clob_get_markets_impl,
    "clob_get_simplified_markets": _clob_get_simplified_markets_impl,
    "clob_get_market": _clob_get_market_impl,
    "clob_create_limit_order": _clob_create_limit_order_impl,
    "clob_create_market_order": _clob_create_market_order_impl,
    "clob_cancel_order": _clob_cancel_order_impl,
    "clob_cancel_all_orders": _clob_cancel_all_orders_impl,
    "clob_get_open_orders": _clob_get_open_orders_impl,
    "clob_get_order": _clob_get_order_impl,
    "clob_get_trades": _clob_get_trades_impl,
    "clob_get_balance_allowance": _clob_get_balance_allowance_impl,
    "clob_get_api_keys": _clob_get_api_keys_impl,
    "clob_create_rfq_request": _clob_create_rfq_request_impl,
    "clob_get_rfq_requests": _clob_get_rfq_requests_impl,
    "clob_create_rfq_quote": _clob_create_rfq_quote_impl,
    "clob_get_rfq_quotes": _clob_get_rfq_quotes_impl,
    "clob_accept_rfq_quote": _clob_accept_rfq_quote_impl,
    "clob_cancel_rfq_request": _clob_cancel_rfq_request_impl,
}


def get_clob_tool_functions() -> Dict[str, Callable]:
    """Get raw CLOB tool callables for non-LangChain configurations."""
    return dict(_CLOB_TOOL_FUNCTIONS)


# =============================================================================
# TOOL COLLECTION FUNCTIONS
# =============================================================================


def get_clob_market_tools() -> List:
    """Get CLOB market data tools (read-only, no auth required)."""
    return [
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
    ]


def get_clob_trading_tools() -> List:
    """Get CLOB trading tools (requires authentication)."""
    return [
        clob_create_limit_order,
        clob_create_market_order,
        clob_cancel_order,
        clob_cancel_all_orders,
        clob_get_open_orders,
        clob_get_order,
        clob_get_trades,
    ]


def get_clob_account_tools() -> List:
    """Get CLOB account tools (requires authentication)."""
    return [
        clob_get_balance_allowance,
        clob_get_api_keys,
    ]


def get_clob_rfq_tools() -> List:
    """Get CLOB RFQ tools (requires authentication)."""
    return [
        clob_create_rfq_request,
        clob_get_rfq_requests,
        clob_create_rfq_quote,
        clob_get_rfq_quotes,
        clob_accept_rfq_quote,
        clob_cancel_rfq_request,
    ]


def get_all_clob_tools() -> List:
    """Get all CLOB tools."""
    return (
        get_clob_market_tools()
        + get_clob_trading_tools()
        + get_clob_account_tools()
        + get_clob_rfq_tools()
    )


def get_clob_readonly_tools() -> List:
    """Get all read-only CLOB tools (no trading)."""
    return get_clob_market_tools()


# =============================================================================
# ARGUMENT REFERENCE
# =============================================================================

CLOB_ARGUMENT_REFERENCE = """
CLOB TOOLS - ARGUMENT REFERENCE (py_clob_client)
=================================================

MARKET DATA TOOLS (No Auth Required)
------------------------------------

clob_health_check() -> No arguments

clob_get_server_time() -> No arguments

clob_get_midpoint(token_id: str)
    token_id: Conditional token ID. Required.

clob_get_price(token_id: str, side: str)
    token_id: Conditional token ID. Required.
    side: "BUY" or "SELL". Required.

clob_get_orderbook(token_id: str)
    token_id: Conditional token ID. Required.

clob_get_spread(token_id: str)
    token_id: Conditional token ID. Required.

clob_get_last_trade_price(token_id: str)
    token_id: Conditional token ID. Required.

clob_get_markets(next_cursor: str = "")
    next_cursor: Pagination cursor. Optional.

clob_get_simplified_markets(next_cursor: str = "")
    next_cursor: Pagination cursor. Optional.

clob_get_market(condition_id: str)
    condition_id: Market condition ID. Required.


TRADING TOOLS (Requires Auth)
-----------------------------

clob_create_limit_order(token_id, price, size, side)
    token_id: Conditional token ID. Required.
    price: 0.01 to 0.99. Required.
    size: Number of shares > 0. Required.
    side: "BUY" or "SELL". Required.

clob_create_market_order(token_id, amount, side)
    token_id: Conditional token ID. Required.
    amount: USDC (BUY) or shares (SELL). Required.
    side: "BUY" or "SELL". Required.

clob_cancel_order(order_id: str)
    order_id: Order hash (0x...). Required.

clob_cancel_all_orders() -> No arguments

clob_get_open_orders(market: str = "", asset_id: str = "")
    market: Filter by market condition ID. Optional.
    asset_id: Filter by token ID. Optional.

clob_get_order(order_id: str)
    order_id: Order hash. Required.

clob_get_trades(market: str = "", maker_address: str = "")
    market: Filter by market. Optional.
    maker_address: Filter by maker. Optional.


ACCOUNT TOOLS (Requires Auth)
-----------------------------

clob_get_balance_allowance(asset_type: str = "COLLATERAL", token_id: str = "")
    asset_type: "COLLATERAL" or "CONDITIONAL". Default: COLLATERAL.
    token_id: Required if asset_type is CONDITIONAL.

clob_get_api_keys() -> No arguments


RFQ TOOLS (Requires Auth)
-------------------------

clob_create_rfq_request(token_id, price, size, side)
    token_id: Conditional token ID. Required.
    price: Desired price. Required.
    size: Number of shares. Required.
    side: "BUY" or "SELL". Required.

clob_get_rfq_requests(state: str = "active", limit: int = 10)
    state: "active" or "inactive". Default: active.
    limit: Max results. Default: 10.

clob_create_rfq_quote(request_id, token_id, price, size, side)
    request_id: RFQ request ID. Required.
    token_id: Conditional token ID. Required.
    price: Quoted price. Required.
    size: Number of shares. Required.
    side: "BUY" or "SELL". Required.

clob_get_rfq_quotes(request_id: str = "", state: str = "", limit: int = 10)
    request_id: Filter by request. Optional.
    state: Filter by state. Optional.
    limit: Max results. Default: 10.

clob_accept_rfq_quote(request_id, quote_id, expiration_seconds: int = 3600)
    request_id: RFQ request ID. Required.
    quote_id: Quote ID to accept. Required.
    expiration_seconds: Order expiry. Default: 3600 (1 hour).

clob_cancel_rfq_request(request_id: str)
    request_id: RFQ request ID. Required.
"""


def print_clob_argument_reference():
    """Print the CLOB tools argument reference."""
    print(CLOB_ARGUMENT_REFERENCE)

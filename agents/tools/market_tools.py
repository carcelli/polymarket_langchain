from typing import List, Dict

from agents.tooling import wrap_tool
from agents.polymarket.gamma import GammaMarketClient

# Initialize the client once (connection pooling)
gamma_client = GammaMarketClient()

def _fetch_active_markets_impl(limit: int = 5) -> List[Dict]:
    """
    Fetches a list of the most active markets on Polymarket.
    Use this to scan for new trading opportunities.
    Returns a list of dictionaries containing market question, description, and volume.
    """
    try:
        # We use get_markets with parse_pydantic=True to get Market objects
        # We filter for active markets directly here
        markets = gamma_client.get_markets(
            querystring_params={
                "active": True,
                "closed": False,
                "archived": False,
                "limit": limit
            },
            parse_pydantic=True
        )
        
        # We serialize back to dict for the LLM tool output
        return [m.dict() for m in markets]
    except Exception as e:
        return [{"error": f"Failed to fetch markets: {str(e)}"}]

def _get_market_details_impl(token_id: str) -> Dict:
    """
    Fetches deep details for a specific market using its Token ID.
    Use this when you need to know the specific Outcome Prices or Spread.
    """
    # For now, we reuse get_active_markets logic, but in production 
    # this would hit the specific /markets/{id} endpoint.
    # We mock this specifically for the bootstrapping phase.
    return {"id": token_id, "status": "active", "spread": 0.01}


fetch_active_markets = wrap_tool(
    _fetch_active_markets_impl,
    name="fetch_active_markets",
)
get_market_details = wrap_tool(
    _get_market_details_impl,
    name="get_market_details",
)

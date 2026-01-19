import json
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Modular Imports
from polymarket_agents.utils.database import fetch_market_metadata, fetch_price_history_raw
from polymarket_agents.utils.analytics import calculate_price_trend

class GetMarketHistoryArgs(BaseModel):
    market_id: str = Field(
        ..., 
        description="The unique Polymarket ID (e.g., '0x...'). Do not use the slug."
    )
    days_back: Optional[int] = Field(
        default=30, 
        description="Days of history to analyze. Defaults to 30."
    )

@tool("market_get_history", args_schema=GetMarketHistoryArgs)
def get_market_history(market_id: str, days_back: int = 30) -> str:
    """
    Retrieves historical price/volume data and analyzes trends.
    Use this to determine market momentum before placing a trade.
    """
    # 1. Fetch Metadata (Guard clause)
    metadata = fetch_market_metadata(market_id)
    if not metadata:
        return json.dumps({
            "error": "Market not found",
            "market_id": market_id,
            "status": "failed"
        })

    # 2. Fetch History (Data Access)
    try:
        raw_rows = fetch_price_history_raw(market_id, days_back)
    except Exception as e:
        return json.dumps({"error": f"Database read failed: {str(e)}"})

    # 3. Format Data (Transformation)
    formatted_history = []
    for row in raw_rows:
        date, yes, no, vol = row
        formatted_history.append({
            "date": date,
            "yes_price": yes,
            "no_price": no,
            "volume": vol,
            # Explicit semantics for the LLM
            "implied_probability": yes 
        })

    # 4. Apply Business Logic (Analytics)
    trend = calculate_price_trend(formatted_history)

    # 5. Construct Final Response
    response = {
        "meta": {
            "id": market_id,
            "days_analyzed": days_back,
            "data_points": len(formatted_history)
        },
        "market_status": {
            "active": metadata["is_active"],
            "closed": metadata["is_closed"],
            "end_date": metadata["end_date"]
        },
        "analysis": {
            "trend": trend,
            "latest_price": formatted_history[-1] if formatted_history else None
        },
        "history": formatted_history
    }

    return json.dumps(response, indent=2)

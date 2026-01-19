import json
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Modular Imports
from polymarket_agents.utils.database import fetch_market_metadata, get_price_stream, PricePoint
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

    # 2. Consume Generator (Data Access)
    # TEXTBOOK CONCEPT: List Comprehension + Generator Consumption
    # We stay lazy as long as possible, only materializing the list here.
    try:
        price_points = [p for p in get_price_stream(market_id, days_back)]
    except Exception as e:
        return json.dumps({"error": f"Database read failed: {str(e)}"})

    # 3. Apply Business Logic (Analytics)
    # Operates on the list of NamedTuples for performance and type safety.
    trend = calculate_price_trend(price_points)

    # 4. Format for JSON (Transformation)
    # TEXTBOOK CONCEPT: List Comprehension with _asdict()
    formatted_history = [
        {
            **p._asdict(),
            "implied_probability": p.yes_price # Explicit semantic for LLM
        }
        for p in price_points
    ]

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
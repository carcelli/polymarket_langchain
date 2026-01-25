import json
from typing import Optional, Type, List
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# Modular Imports
from polymarket_agents.utils.database import (
    fetch_market_metadata,
    get_price_stream,
    PricePoint,
)
from polymarket_agents.utils.analytics import calculate_price_trend
from polymarket_agents.utils.charfinder_client import query_unicode_names


class GetMarketHistoryArgs(BaseModel):
    market_id: str = Field(
        ...,
        description="The unique Polymarket ID (e.g., '0x...'). Do not use the slug.",
    )
    days_back: Optional[int] = Field(
        default=30, description="Days of history to analyze. Defaults to 30."
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
        return json.dumps(
            {"error": "Market not found", "market_id": market_id, "status": "failed"}
        )

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
        {**p._asdict(), "implied_probability": p.yes_price}  # Explicit semantic for LLM
        for p in price_points
    ]

    # 5. Construct Final Response
    response = {
        "meta": {
            "id": market_id,
            "days_analyzed": days_back,
            "data_points": len(formatted_history),
        },
        "market_status": {
            "active": metadata["is_active"],
            "closed": metadata["is_closed"],
            "end_date": metadata["end_date"],
        },
        "analysis": {
            "trend": trend,
            "latest_price": formatted_history[-1] if formatted_history else None,
        },
        "history": formatted_history,
    }

    return json.dumps(response, indent=2)


class FindUnicodeCharsArgs(BaseModel):
    search_term: str = Field(
        ...,
        description="Term to search for in Unicode character names (e.g., 'chess', 'arrow', 'greek')",
    )
    max_results: Optional[int] = Field(
        default=10, description="Maximum number of results to return. Defaults to 10."
    )


@tool("find_unicode_chars", args_schema=FindUnicodeCharsArgs)
def find_unicode_chars(search_term: str, max_results: int = 10) -> str:
    """
    Searches Unicode character names for symbols matching the search term.

    Useful for:
    - Finding symbols for market descriptions (e.g., 'bitcoin' → ₿, 'euro' → €)
    - Ontology mapping and metadata enrichment
    - Converting text symbols to Unicode characters

    The character finder server must be running on localhost:2323.
    Start it with: python -m polymarket_agents.utils.charfinder_server
    """
    try:
        results = query_unicode_names(search_term)

        # Limit results
        limited_results = results[:max_results]

        # Parse results into structured format
        parsed_results = []
        for result in limited_results:
            try:
                parts = result.split("\t")
                if len(parts) >= 3:
                    code, char, name = parts[0], parts[1], parts[2]
                    parsed_results.append(
                        {"unicode_code": code, "character": char, "name": name}
                    )
            except (ValueError, IndexError):
                # Skip malformed results
                continue

        response = {
            "search_term": search_term,
            "total_found": len(results),
            "results_returned": len(parsed_results),
            "characters": parsed_results,
        }

        if not results:
            response["message"] = (
                f"No Unicode characters found matching '{search_term}'"
            )

        return json.dumps(response, indent=2, ensure_ascii=False)

    except ConnectionError:
        return json.dumps(
            {
                "error": "Character finder server not available",
                "message": "Start the server with: python -m polymarket_agents.utils.charfinder_server",
                "search_term": search_term,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {"error": f"Unicode search failed: {str(e)}", "search_term": search_term},
            indent=2,
        )

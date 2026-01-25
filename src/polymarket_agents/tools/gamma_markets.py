"""
Gamma Markets Tool for Polymarket Integration

Provides read-only access to Polymarket markets via the Gamma API.
Returns structured market snapshots with implied probabilities, volume, and metadata.
"""

from __future__ import annotations

import json
import httpx
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class GammaMarketsInput(BaseModel):
    """Input schema for Gamma markets API queries."""

    active: Optional[bool] = Field(True, description="Only return open/active markets.")
    limit: Optional[int] = Field(
        50, ge=1, le=100, description="Max markets to return (Gamma caps ~100)."
    )
    question_contains: Optional[str] = Field(
        None, description="Substring filter on market question/title."
    )


class GammaMarketsTool(BaseTool):
    """LangChain tool for fetching Polymarket markets via Gamma API."""

    name: str = "gamma_fetch_markets"
    description: str = (
        "Fetch live Polymarket markets via Gamma API. "
        "Returns structured snapshots with implied YES probability, volume, and metadata. "
        "Read-only and rate-limit safe for market discovery and analysis."
    )
    args_schema = GammaMarketsInput

    def _run(
        self,
        active: bool = True,
        limit: int = 50,
        question_contains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch markets from Gamma API and normalize to structured format.

        Args:
            active: Only return open/active markets
            limit: Maximum number of markets to return
            question_contains: Filter markets by question substring

        Returns:
            List of normalized market snapshots, sorted by volume descending
        """
        url = "https://gamma-api.polymarket.com/markets"
        params = {"active": str(active).lower(), "limit": min(limit, 100)}
        if question_contains:
            params["question_contains"] = question_contains.lower()

        try:
            with httpx.Client(timeout=10.0) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                raw_data = resp.json()

            # Handle different response formats
            if isinstance(raw_data, list):
                markets_data = raw_data
            elif isinstance(raw_data, dict):
                markets_data = raw_data.get("data", [])
                if isinstance(markets_data, dict):
                    markets_data = [markets_data]
                elif not isinstance(markets_data, list):
                    markets_data = []
            else:
                markets_data = []

            # Normalize to agent-friendly schema
            markets = []
            for m in markets_data[:limit]:
                try:
                    # Extract yes probability (implied market probability)
                    yes_price = 0.5  # Default fallback
                    if "yes_price" in m and m["yes_price"] is not None:
                        yes_price = float(m["yes_price"])
                    elif "outcomePrices" in m and m["outcomePrices"]:
                        # outcomePrices comes as a JSON string array like "[\"0.6\", \"0.4\"]"
                        try:
                            if isinstance(m["outcomePrices"], str):
                                prices = json.loads(m["outcomePrices"])
                            else:
                                prices = m["outcomePrices"]

                            if isinstance(prices, list) and len(prices) > 0:
                                yes_price = float(prices[0])
                        except (json.JSONDecodeError, ValueError, IndexError):
                            # Keep default fallback
                            pass

                    # Ensure valid probability range
                    yes_price = max(0.01, min(0.99, yes_price))

                    market_snapshot = {
                        "question": str(m.get("question", "")),
                        "slug": str(m.get("slug", "")),
                        "yes_prob": yes_price,
                        "no_prob": round(1 - yes_price, 4),
                        "volume": float(m.get("volume", 0)),
                        "liquidity": float(m.get("liquidity", 0)),
                        "end_date": m.get("end_date"),
                        "active": m.get("active", True),
                    }
                    markets.append(market_snapshot)

                except (KeyError, ValueError, TypeError) as e:
                    # Skip malformed market data but continue processing
                    continue

            # Sort by volume descending for most relevant markets first
            markets.sort(key=lambda x: x["volume"], reverse=True)

            return markets

        except httpx.TimeoutException:
            return [{"error": "Request timeout - Gamma API may be slow"}]
        except httpx.HTTPStatusError as e:
            return [
                {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
            ]
        except Exception as e:
            return [{"error": f"Unexpected error: {str(e)}"}]

    def get_markets_with_edge(
        self, limit: int = 20, min_volume: float = 1000, edge_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Get markets with potential edge - high volume and probabilities away from 50%.

        Edge score = volume × |0.5 - yes_prob|
        Higher scores indicate markets where crowd has strong conviction.

        Args:
            limit: Maximum markets to return
            min_volume: Minimum volume threshold
            edge_threshold: Minimum |0.5 - prob| for consideration

        Returns:
            Markets sorted by edge score descending
        """
        all_markets = self._run(active=True, limit=100)  # Get more to filter

        if (
            not all_markets
            or isinstance(all_markets[0], dict)
            and "error" in all_markets[0]
        ):
            return all_markets

        # Calculate edge scores and filter
        markets_with_edge = []
        for market in all_markets:
            if isinstance(market, dict) and "error" not in market:
                volume = market.get("volume", 0)
                yes_prob = market.get("yes_prob", 0.5)

                # Skip low volume markets
                if volume < min_volume:
                    continue

                # Calculate edge: how far from 50/50
                edge = abs(0.5 - yes_prob)

                # Skip markets too close to 50/50
                if edge < edge_threshold:
                    continue

                # Edge score: volume × edge (higher = more interesting)
                edge_score = volume * edge

                market["edge_score"] = edge_score
                market["edge"] = edge
                markets_with_edge.append(market)

        # Sort by edge score descending
        markets_with_edge.sort(key=lambda x: x["edge_score"], reverse=True)

        return markets_with_edge[:limit]


# Standalone usage for testing
if __name__ == "__main__":
    tool = GammaMarketsTool()
    result = tool._run(active=True, limit=5, question_contains="election")
    print("Sample markets:")
    for market in result[:3]:
        print(
            f"  {market.get('question', 'N/A')[:60]}... | Yes: {market.get('yes_prob', 'N/A'):.1%} | Vol: ${market.get('volume', 0):,.0f}"
        )

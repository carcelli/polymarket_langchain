from pydantic import BaseModel
from typing import List, Optional

class Market(BaseModel):
    """Legacy market model for existing codebase compatibility."""
    id: str
    question: str
    outcomes: List[str]
    volume: float
    spread: float
    # Add other fields used in graph/state.py

class MarketSnapshot(BaseModel):
    """Structured snapshot of Polymarket data for agent consumption.

    Optimized for read-only analysis with implied probabilities
    and key trading metrics for market discovery and ML features.
    """
    question: str
    slug: str
    yes_prob: float  # Implied probability of YES outcome (0.0-1.0)
    no_prob: float   # Implied probability of NO outcome (0.0-1.0)
    volume: float    # Trading volume in USD
    liquidity: float # Market liquidity score
    end_date: Optional[str] = None
    active: bool = True
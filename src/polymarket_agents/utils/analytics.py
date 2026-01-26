from typing import Sequence
from polymarket_agents.utils.database import PricePoint


def calculate_price_trend(
    history: Sequence[PricePoint], threshold: float = 0.05
) -> str:
    """
    Determines the trend direction based on first and last price points.

    Args:
        history: Sequence of PricePoint objects
        threshold: Percentage change required to declare a trend (default 5%)
    """
    if len(history) < 2:
        return "insufficient_data"

    # TEXTBOOK CONCEPT: Tuple Unpacking
    # Accessing attributes from NamedTuple PricePoint
    first_price = history[0].yes_price
    last_price = history[-1].yes_price

    if first_price is None or last_price is None:
        return "insufficient_data"

    # Avoid division by zero
    if first_price == 0:
        return "stable"

    pct_change = (last_price - first_price) / first_price

    if pct_change > threshold:
        return "rising"
    elif pct_change < -threshold:
        return "falling"
    else:
        return "stable"

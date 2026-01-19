from typing import List, Dict, Any

def calculate_price_trend(history: List[Dict[str, Any]], threshold: float = 0.05) -> str:
    """
    Determines the trend direction based on first and last price points.
    
    Args:
        history: List of price dicts containing 'yes_price'
        threshold: Percentage change required to declare a trend (default 5%)
    """
    if len(history) < 2:
        return "insufficient_data"
    
    # Handle potential None values safely
    first_price = history[0].get("yes_price")
    last_price = history[-1].get("yes_price")
    
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

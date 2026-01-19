import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class PricePoint:
    yes_price: float
    volume: float
    timestamp: float

def get_price_stream(market_id: str, days_back: int = 90) -> List[PricePoint]:
    """
    Simulate fetching price stream for a market.
    In a real system, this would query the SQLite DB or an API.
    For now, we generate a synthetic random walk + volume spikes.
    """
    # Create deterministic random stream based on market_id
    seed = hash(market_id) % 2**32
    rng = np.random.RandomState(seed)
    
    steps = days_back
    # Random walk for price
    price = 0.5
    prices = []
    
    # Random volume
    base_volume = 1000.0
    volumes = []
    
    for _ in range(steps):
        # Price drift
        drift = rng.normal(0, 0.05)
        price += drift
        price = np.clip(price, 0.01, 0.99)
        prices.append(price)
        
        # Volume with occasional spikes
        vol_noise = rng.lognormal(0, 1.0)
        vol = base_volume * vol_noise
        if rng.rand() < 0.05: # 5% chance of spike
            vol *= 5.0
        volumes.append(vol)
    
    history = []
    for i in range(steps):
        history.append(PricePoint(
            yes_price=prices[i],
            volume=volumes[i],
            timestamp=i # Simplified
        ))
        
    return history

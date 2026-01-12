from pydantic import BaseModel
from typing import List

class Market(BaseModel):
    id: str
    question: str
    outcomes: List[str]
    volume: float
    spread: float
    # Add other fields used in graph/state.py
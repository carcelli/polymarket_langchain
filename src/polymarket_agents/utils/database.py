import sqlite3
import json
from typing import Optional, List, Dict, Any, Generator, NamedTuple
from datetime import datetime, timedelta
from polymarket_agents.config import DATABASE_PATH

# TEXTBOOK CONCEPT: NamedTuple
# Lightweight, immutable record. More readable than row[1], faster than a Dict.
class PricePoint(NamedTuple):
    date: str
    yes_price: float
    no_price: float
    volume: float

def get_db_connection():
    """Context manager factory for database connections."""
    return sqlite3.connect(str(DATABASE_PATH))

def fetch_market_metadata(market_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves static metadata for a specific market."""
    query = """
        SELECT active, closed, marketType, outcomes, endDate
        FROM markets
        WHERE id = ?
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, (market_id,))
            row = cursor.fetchone()
            if not row:
                return None
            active, closed, market_type, outcomes_raw, end_date = row
            try:
                outcomes = json.loads(outcomes_raw) if outcomes_raw else []
            except json.JSONDecodeError:
                outcomes = []
            return {
                "is_active": bool(active),
                "is_closed": bool(closed),
                "market_type": market_type,
                "outcomes": outcomes,
                "end_date": end_date
            }
    except sqlite3.Error as e:
        print(f"DB Error: {e}")
        return None

# TEXTBOOK CONCEPT: Generator
# This acts as a lazy pipeline. It doesn't load 10,000 prices into RAM.
def get_price_stream(market_id: str, days_back: int = 30) -> Generator[PricePoint, None, None]:
    """
    Yields PricePoint records one by one from the database.
    """
    cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    query = """
        SELECT date, yes_price, no_price, volume
        FROM price_history
        WHERE market_id = ? AND date >= ?
        ORDER BY date ASC
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (market_id, cutoff_date))
        for row in cursor:
            # TEXTBOOK CONCEPT: Tuple Unpacking/Mapping
            yield PricePoint(*row)
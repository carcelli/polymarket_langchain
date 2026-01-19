import sqlite3
import json
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime, timedelta
from polymarket_agents.config import DATABASE_PATH

def get_db_connection():
    """Context manager factory for database connections."""
    return sqlite3.connect(str(DATABASE_PATH))

def fetch_market_metadata(market_id: str) -> Optional[Dict[str, Any]]:
    """
    Retrieves static metadata for a specific market.
    Returns None if market not found.
    """
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
            
            # Safe JSON parsing
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
        # Log error here if you have a logger
        print(f"DB Error in fetch_market_metadata: {e}")
        return None

def fetch_price_history_raw(market_id: str, days_back: int) -> List[Tuple]:
    """
    Fetches raw price history rows from the database.
    Separates SQL concern from formatting concern.
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
        return cursor.fetchall()

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

class MemoryManager:
    """
    Manages structured memory (SQLite) and semantic memory (Vector DB - placeholder/integration).
    """
    
    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Markets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                id TEXT PRIMARY KEY,
                question TEXT,
                description TEXT,
                outcomes TEXT,
                outcome_prices TEXT,
                volume REAL,
                liquidity REAL,
                active BOOLEAN,
                end_date TEXT,
                last_updated TEXT
            )
        """)
        
        # News table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT,
                title TEXT,
                description TEXT,
                url TEXT,
                published_at TEXT,
                ingested_at TEXT,
                FOREIGN KEY(market_id) REFERENCES markets(id)
            )
        """)
        
        conn.commit()
        conn.close()

    def add_market(self, market_data: Dict[str, Any], news_data: List[Dict[str, Any]] = None):
        """Add or update a market and its associated news."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Parse/Format fields
        m_id = str(market_data.get('id'))
        question = market_data.get('question')
        description = market_data.get('description', '')
        
        # Handle list/dict fields by dumping to JSON
        outcomes = json.dumps(market_data.get('outcomes', []))
        outcome_prices = json.dumps(market_data.get('outcome_prices', []))
        
        # Numeric fields
        volume = float(market_data.get('volume', 0) or 0)
        liquidity = float(market_data.get('liquidity', 0) or 0)
        
        active = 1 if market_data.get('active') else 0
        end_date = market_data.get('endDate') or market_data.get('end')
        last_updated = datetime.now().isoformat()

        # Upsert Market
        cursor.execute("""
            INSERT OR REPLACE INTO markets 
            (id, question, description, outcomes, outcome_prices, volume, liquidity, active, end_date, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (m_id, question, description, outcomes, outcome_prices, volume, liquidity, active, end_date, last_updated))
        
        # Insert News
        if news_data:
            for article in news_data:
                # Check duplication by URL/Title if needed, for now just insert
                cursor.execute("""
                    INSERT INTO news (market_id, title, description, url, published_at, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    m_id,
                    article.get('title'),
                    article.get('description'),
                    article.get('url'),
                    article.get('published_at') or article.get('publishedAt'),
                    last_updated
                ))

        conn.commit()
        conn.close()
        print(f"Memory: Stored market {m_id} ('{question}') with {len(news_data or [])} articles.")

    def get_market(self, market_id: str) -> Optional[Dict]:
        """Retrieve a market by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM markets WHERE id = ?", (str(market_id),))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            d = dict(row)
            d['outcomes'] = json.loads(d['outcomes'])
            d['outcome_prices'] = json.loads(d['outcome_prices'])
            d['active'] = bool(d['active'])
            return d
        return None

    def get_market_news(self, market_id: str) -> List[Dict]:
        """Retrieve news for a market."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM news WHERE market_id = ?", (str(market_id),))
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(r) for r in rows]

    def list_recent_markets(self, limit: int = 10) -> List[Dict]:
        """List recently updated markets."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM markets ORDER BY last_updated DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(r) for r in rows]

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any

class MemoryManager:
    """
    Manages structured memory (SQLite) for Polymarket data.
    
    Uses data/markets.db as the primary database for categorized market data.
    Provides methods for agents to query markets by category, volume, etc.
    """
    
    def __init__(self, db_path: str = "data/markets.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Markets table - comprehensive schema for categorized markets
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                id TEXT PRIMARY KEY,
                question TEXT,
                description TEXT,
                category TEXT,
                outcomes TEXT,
                outcome_prices TEXT,
                volume REAL,
                liquidity REAL,
                active BOOLEAN,
                end_date TEXT,
                slug TEXT,
                clob_token_ids TEXT,
                event_id TEXT,
                tags TEXT,
                last_updated TEXT
            )
        """)
        
        # Create indexes for efficient querying
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON markets(category)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_active ON markets(active)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_volume ON markets(volume)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_end_date ON markets(end_date)")
        
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
        
        # Price history table - track price movements over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS price_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                yes_price REAL,
                no_price REAL,
                volume REAL,
                liquidity REAL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY(market_id) REFERENCES markets(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_market ON price_history(market_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_history(timestamp)")
        
        # Bets tracking table - user positions and P&L
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT NOT NULL,
                market_question TEXT,
                side TEXT NOT NULL,  -- 'YES' or 'NO'
                entry_price REAL NOT NULL,
                current_price REAL,
                shares REAL NOT NULL,
                cost_basis REAL NOT NULL,
                current_value REAL,
                unrealized_pnl REAL,
                realized_pnl REAL DEFAULT 0,
                status TEXT DEFAULT 'OPEN',  -- OPEN, CLOSED, WON, LOST
                entry_date TEXT NOT NULL,
                exit_date TEXT,
                exit_price REAL,
                notes TEXT,
                strategy TEXT,  -- e.g., 'value_bet', 'momentum', 'hedge'
                confidence REAL,  -- 0-1 confidence at entry
                FOREIGN KEY(market_id) REFERENCES markets(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bets_market ON bets(market_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status)")
        
        # Research table - store gathered information for markets
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS research (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT,
                research_type TEXT,  -- 'news', 'analysis', 'data_point', 'sentiment'
                source TEXT,
                content TEXT,
                sentiment_score REAL,  -- -1 to 1 (bearish to bullish)
                confidence REAL,  -- 0 to 1
                created_at TEXT,
                expires_at TEXT,
                FOREIGN KEY(market_id) REFERENCES markets(id)
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_research_market ON research(market_id)")
        
        # Market analytics table - computed stats for betting
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_analytics (
                market_id TEXT PRIMARY KEY,
                implied_prob_yes REAL,  -- Market implied probability
                estimated_prob_yes REAL,  -- Our estimated true probability
                edge REAL,  -- estimated_prob - implied_prob (positive = value)
                expected_value REAL,  -- EV of YES bet
                kelly_fraction REAL,  -- Kelly criterion suggested bet size
                volatility REAL,  -- Price volatility (std dev)
                price_momentum REAL,  -- Recent price trend
                volume_trend REAL,  -- Volume momentum
                last_analysis TEXT,  -- Timestamp of last analysis
                analyst_notes TEXT,
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
        category = market_data.get('category', 'other')
        
        # Handle list/dict fields by dumping to JSON
        outcomes = market_data.get('outcomes', [])
        if isinstance(outcomes, str):
            outcomes = outcomes  # Already JSON string
        else:
            outcomes = json.dumps(outcomes)
            
        outcome_prices = market_data.get('outcome_prices') or market_data.get('outcomePrices', [])
        if isinstance(outcome_prices, str):
            outcome_prices = outcome_prices
        else:
            outcome_prices = json.dumps(outcome_prices)
        
        # Numeric fields
        volume = float(market_data.get('volume', 0) or 0)
        liquidity = float(market_data.get('liquidity', 0) or 0)
        
        active = 1 if market_data.get('active') else 0
        end_date = market_data.get('endDate') or market_data.get('end_date') or market_data.get('end')
        slug = market_data.get('slug', '')
        clob_token_ids = market_data.get('clobTokenIds') or market_data.get('clob_token_ids', '')
        event_id = market_data.get('event_id', '')
        
        # Tags handling
        tags = market_data.get('tags', [])
        if isinstance(tags, list):
            tags = json.dumps(tags)
        
        last_updated = datetime.now().isoformat()

        # Upsert Market
        cursor.execute("""
            INSERT OR REPLACE INTO markets 
            (id, question, description, category, outcomes, outcome_prices, volume, liquidity, 
             active, end_date, slug, clob_token_ids, event_id, tags, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (m_id, question, description, category, outcomes, outcome_prices, volume, liquidity, 
              active, end_date, slug, clob_token_ids, event_id, tags, last_updated))
        
        # Insert News
        if news_data:
            for article in news_data:
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

    def _parse_market_row(self, row: sqlite3.Row) -> Dict:
        """Parse a market row into a dictionary with proper types."""
        d = dict(row)
        
        # Parse JSON fields
        try:
            d['outcomes'] = json.loads(d['outcomes']) if d.get('outcomes') else []
        except (json.JSONDecodeError, TypeError):
            d['outcomes'] = []
            
        try:
            d['outcome_prices'] = json.loads(d['outcome_prices']) if d.get('outcome_prices') else []
        except (json.JSONDecodeError, TypeError):
            d['outcome_prices'] = []
            
        try:
            d['tags'] = json.loads(d['tags']) if d.get('tags') else []
        except (json.JSONDecodeError, TypeError):
            d['tags'] = []
        
        d['active'] = bool(d.get('active'))
        return d

    def get_market(self, market_id: str) -> Optional[Dict]:
        """Retrieve a market by ID."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM markets WHERE id = ?", (str(market_id),))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._parse_market_row(row)
        return None

    def get_market_by_slug(self, slug: str) -> Optional[Dict]:
        """Retrieve a market by slug."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM markets WHERE slug = ?", (slug,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._parse_market_row(row)
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
        
        return [self._parse_market_row(r) for r in rows]

    def list_markets_by_category(self, category: str, limit: int = 20, active_only: bool = True) -> List[Dict]:
        """List markets in a specific category."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if active_only:
            cursor.execute(
                "SELECT * FROM markets WHERE category = ? AND active = 1 ORDER BY volume DESC LIMIT ?",
                (category, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM markets WHERE category = ? ORDER BY volume DESC LIMIT ?",
                (category, limit)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._parse_market_row(r) for r in rows]

    def list_top_volume_markets(self, limit: int = 20, category: str = None) -> List[Dict]:
        """List markets with highest volume."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if category:
            cursor.execute(
                "SELECT * FROM markets WHERE category = ? AND active = 1 ORDER BY volume DESC LIMIT ?",
                (category, limit)
            )
        else:
            cursor.execute(
                "SELECT * FROM markets WHERE active = 1 ORDER BY volume DESC LIMIT ?",
                (limit,)
            )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._parse_market_row(r) for r in rows]

    def search_markets(self, query: str, limit: int = 20) -> List[Dict]:
        """Search markets by question text."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT * FROM markets WHERE question LIKE ? ORDER BY volume DESC LIMIT ?",
            (f"%{query}%", limit)
        )
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._parse_market_row(r) for r in rows]

    def get_categories(self) -> List[Dict]:
        """Get all categories with market counts and total volume."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT category, COUNT(*) as count, SUM(volume) as total_volume
            FROM markets 
            WHERE active = 1
            GROUP BY category
            ORDER BY count DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [{"category": r[0], "count": r[1], "total_volume": r[2] or 0} for r in rows]

    def get_stats(self) -> Dict:
        """Get database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM markets WHERE active = 1")
        total_markets = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(volume) FROM markets WHERE active = 1")
        total_volume = cursor.fetchone()[0] or 0
        
        cursor.execute("SELECT COUNT(DISTINCT category) FROM markets")
        total_categories = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_markets": total_markets,
            "total_volume": total_volume,
            "total_categories": total_categories
        }

    def cleanup_expired_markets(self, grace_hours: int = 24) -> Dict[str, Any]:
        """
        Delete markets that have expired.
        
        Markets are deleted if their end_date has passed by more than grace_hours.
        This keeps the database lean and relevant.
        
        Args:
            grace_hours: Hours after expiration before deletion (default: 24)
                         This allows for resolution data to settle.
        
        Returns:
            Dict with cleanup statistics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now()
        
        # Count expired markets before deletion
        cursor.execute("""
            SELECT COUNT(*), SUM(volume)
            FROM markets 
            WHERE end_date IS NOT NULL 
            AND end_date != ''
            AND datetime(end_date) < datetime('now', ?)
        """, (f'-{grace_hours} hours',))
        
        result = cursor.fetchone()
        expired_count = result[0] or 0
        expired_volume = result[1] or 0
        
        # Get category breakdown of expired markets
        cursor.execute("""
            SELECT category, COUNT(*) 
            FROM markets 
            WHERE end_date IS NOT NULL 
            AND end_date != ''
            AND datetime(end_date) < datetime('now', ?)
            GROUP BY category
        """, (f'-{grace_hours} hours',))
        
        expired_by_category = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Delete expired markets
        cursor.execute("""
            DELETE FROM markets 
            WHERE end_date IS NOT NULL 
            AND end_date != ''
            AND datetime(end_date) < datetime('now', ?)
        """, (f'-{grace_hours} hours',))
        
        deleted_count = cursor.rowcount
        
        # Also delete associated news for expired markets
        cursor.execute("""
            DELETE FROM news 
            WHERE market_id NOT IN (SELECT id FROM markets)
        """)
        deleted_news = cursor.rowcount
        
        conn.commit()
        
        # Get remaining stats
        cursor.execute("SELECT COUNT(*) FROM markets")
        remaining_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "deleted_count": deleted_count,
            "deleted_volume": expired_volume,
            "deleted_by_category": expired_by_category,
            "deleted_news": deleted_news,
            "remaining_markets": remaining_count,
            "cleanup_time": now.isoformat(),
            "grace_hours": grace_hours
        }

    def get_expiring_soon(self, hours: int = 24, limit: int = 20) -> List[Dict]:
        """
        Get markets expiring within the specified hours.
        
        Useful for monitoring and last-minute trading opportunities.
        
        Args:
            hours: Hours from now to check
            limit: Max markets to return
            
        Returns:
            List of market dicts expiring soon
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM markets 
            WHERE active = 1 
            AND end_date IS NOT NULL 
            AND end_date != ''
            AND datetime(end_date) > datetime('now')
            AND datetime(end_date) < datetime('now', ?)
            ORDER BY end_date ASC
            LIMIT ?
        """, (f'+{hours} hours', limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._parse_market_row(r) for r in rows]

    # =========================================================================
    # PRICE HISTORY METHODS
    # =========================================================================
    
    def record_price(self, market_id: str, yes_price: float, no_price: float = None,
                     volume: float = None, liquidity: float = None):
        """Record a price snapshot for a market."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if no_price is None:
            no_price = 1.0 - yes_price
        
        cursor.execute("""
            INSERT INTO price_history (market_id, yes_price, no_price, volume, liquidity, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (market_id, yes_price, no_price, volume, liquidity, datetime.now().isoformat()))
        
        conn.commit()
        conn.close()
    
    def get_price_history(self, market_id: str, hours: int = 24) -> List[Dict]:
        """Get price history for a market."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM price_history 
            WHERE market_id = ? 
            AND datetime(timestamp) > datetime('now', ?)
            ORDER BY timestamp ASC
        """, (market_id, f'-{hours} hours'))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(r) for r in rows]
    
    def get_price_change(self, market_id: str, hours: int = 24) -> Optional[Dict]:
        """Calculate price change over time period."""
        history = self.get_price_history(market_id, hours)
        if len(history) < 2:
            return None
        
        first = history[0]
        last = history[-1]
        
        return {
            "market_id": market_id,
            "hours": hours,
            "start_price": first['yes_price'],
            "end_price": last['yes_price'],
            "change": last['yes_price'] - first['yes_price'],
            "change_pct": ((last['yes_price'] - first['yes_price']) / first['yes_price'] * 100) if first['yes_price'] > 0 else 0,
            "data_points": len(history)
        }

    # =========================================================================
    # BETS TRACKING METHODS
    # =========================================================================
    
    def add_bet(self, market_id: str, side: str, entry_price: float, shares: float,
                strategy: str = None, confidence: float = None, notes: str = None) -> int:
        """Record a new bet/position."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get market question for reference
        market = self.get_market(market_id)
        question = market['question'] if market else 'Unknown'
        
        cost_basis = entry_price * shares
        
        cursor.execute("""
            INSERT INTO bets (market_id, market_question, side, entry_price, current_price, 
                            shares, cost_basis, current_value, unrealized_pnl, status,
                            entry_date, strategy, confidence, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?)
        """, (market_id, question, side.upper(), entry_price, entry_price,
              shares, cost_basis, cost_basis, 0.0,
              datetime.now().isoformat(), strategy, confidence, notes))
        
        bet_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return bet_id
    
    def update_bet_prices(self):
        """Update all open bets with current market prices."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all open bets
        cursor.execute("SELECT id, market_id, side, entry_price, shares FROM bets WHERE status = 'OPEN'")
        open_bets = cursor.fetchall()
        
        for bet in open_bets:
            bet_id, market_id, side, entry_price, shares = bet
            market = self.get_market(market_id)
            
            if market and market.get('outcome_prices'):
                prices = market['outcome_prices']
                if len(prices) >= 2:
                    try:
                        current_price = float(prices[0]) if side == 'YES' else float(prices[1])
                        current_value = current_price * shares
                        unrealized_pnl = current_value - (entry_price * shares)
                        
                        cursor.execute("""
                            UPDATE bets SET current_price = ?, current_value = ?, unrealized_pnl = ?
                            WHERE id = ?
                        """, (current_price, current_value, unrealized_pnl, bet_id))
                    except (ValueError, IndexError):
                        pass
        
        conn.commit()
        conn.close()
    
    def close_bet(self, bet_id: int, exit_price: float, status: str = 'CLOSED'):
        """Close a bet with final price."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT shares, entry_price FROM bets WHERE id = ?", (bet_id,))
        row = cursor.fetchone()
        
        if row:
            shares, entry_price = row
            realized_pnl = (exit_price - entry_price) * shares
            
            cursor.execute("""
                UPDATE bets SET exit_price = ?, exit_date = ?, status = ?, 
                              realized_pnl = ?, current_price = ?, 
                              current_value = ?, unrealized_pnl = 0
                WHERE id = ?
            """, (exit_price, datetime.now().isoformat(), status, realized_pnl,
                  exit_price, exit_price * shares, bet_id))
        
        conn.commit()
        conn.close()
    
    def get_open_bets(self) -> List[Dict]:
        """Get all open bets."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM bets WHERE status = 'OPEN' ORDER BY entry_date DESC")
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(r) for r in rows]
    
    def get_bet_history(self, limit: int = 50) -> List[Dict]:
        """Get all bets (open and closed)."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM bets ORDER BY entry_date DESC LIMIT ?", (limit,))
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(r) for r in rows]
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary stats."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Open positions
        cursor.execute("""
            SELECT COUNT(*), SUM(cost_basis), SUM(current_value), SUM(unrealized_pnl)
            FROM bets WHERE status = 'OPEN'
        """)
        open_row = cursor.fetchone()
        
        # Closed positions
        cursor.execute("""
            SELECT COUNT(*), SUM(realized_pnl) FROM bets WHERE status != 'OPEN'
        """)
        closed_row = cursor.fetchone()
        
        conn.close()
        
        return {
            "open_positions": open_row[0] or 0,
            "total_invested": open_row[1] or 0,
            "current_value": open_row[2] or 0,
            "unrealized_pnl": open_row[3] or 0,
            "closed_positions": closed_row[0] or 0,
            "realized_pnl": closed_row[1] or 0,
            "total_pnl": (open_row[3] or 0) + (closed_row[1] or 0)
        }

    # =========================================================================
    # RESEARCH METHODS
    # =========================================================================
    
    def add_research(self, market_id: str, research_type: str, content: str,
                     source: str = None, sentiment_score: float = None,
                     confidence: float = None, expires_hours: int = 24) -> int:
        """Add research/intelligence for a market."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        expires_at = None
        if expires_hours:
            from datetime import timedelta
            expires_at = (datetime.now() + timedelta(hours=expires_hours)).isoformat()
        
        cursor.execute("""
            INSERT INTO research (market_id, research_type, source, content, 
                                sentiment_score, confidence, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (market_id, research_type, source, content, sentiment_score,
              confidence, datetime.now().isoformat(), expires_at))
        
        research_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return research_id
    
    def get_market_research(self, market_id: str, include_expired: bool = False) -> List[Dict]:
        """Get all research for a market."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if include_expired:
            cursor.execute("""
                SELECT * FROM research WHERE market_id = ? ORDER BY created_at DESC
            """, (market_id,))
        else:
            cursor.execute("""
                SELECT * FROM research WHERE market_id = ? 
                AND (expires_at IS NULL OR datetime(expires_at) > datetime('now'))
                ORDER BY created_at DESC
            """, (market_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(r) for r in rows]

    # =========================================================================
    # MARKET ANALYTICS METHODS
    # =========================================================================
    
    def update_market_analytics(self, market_id: str, estimated_prob: float = None,
                                 analyst_notes: str = None) -> Dict:
        """
        Update analytics for a market (calculates edge, EV, Kelly).
        
        Args:
            market_id: Market to analyze
            estimated_prob: Your estimated true probability (0-1)
            analyst_notes: Optional notes
        
        Returns:
            Computed analytics dict
        """
        market = self.get_market(market_id)
        if not market:
            return {"error": "Market not found"}
        
        prices = market.get('outcome_prices', [])
        if len(prices) < 2:
            return {"error": "No price data"}
        
        try:
            yes_price = float(prices[0])
            implied_prob = yes_price  # Market price = implied probability
            
            analytics = {
                "market_id": market_id,
                "implied_prob_yes": implied_prob,
                "estimated_prob_yes": estimated_prob,
                "edge": None,
                "expected_value": None,
                "kelly_fraction": None,
                "volatility": None,
                "price_momentum": None,
                "last_analysis": datetime.now().isoformat(),
                "analyst_notes": analyst_notes
            }
            
            if estimated_prob is not None:
                # Edge = our estimate - market price
                edge = estimated_prob - implied_prob
                analytics["edge"] = edge
                
                # Expected Value: EV = (prob * payout) - (1-prob) * stake
                # For YES bet at price p, payout if win = 1/p, stake = 1
                # EV = estimated_prob * (1 - yes_price) - (1 - estimated_prob) * yes_price
                ev = estimated_prob * (1 - yes_price) - (1 - estimated_prob) * yes_price
                analytics["expected_value"] = ev
                
                # Kelly Criterion: f* = (bp - q) / b
                # where b = odds = (1-p)/p, p = our prob, q = 1-p
                if yes_price > 0 and yes_price < 1:
                    b = (1 - yes_price) / yes_price  # Decimal odds minus 1
                    kelly = (b * estimated_prob - (1 - estimated_prob)) / b
                    kelly = max(0, min(kelly, 0.25))  # Cap at 25% of bankroll
                    analytics["kelly_fraction"] = kelly
            
            # Calculate price momentum if we have history
            price_change = self.get_price_change(market_id, hours=24)
            if price_change:
                analytics["price_momentum"] = price_change["change"]
            
            # Store analytics
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO market_analytics 
                (market_id, implied_prob_yes, estimated_prob_yes, edge, expected_value,
                 kelly_fraction, volatility, price_momentum, last_analysis, analyst_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (market_id, analytics["implied_prob_yes"], analytics["estimated_prob_yes"],
                  analytics["edge"], analytics["expected_value"], analytics["kelly_fraction"],
                  analytics["volatility"], analytics["price_momentum"],
                  analytics["last_analysis"], analytics["analyst_notes"]))
            
            conn.commit()
            conn.close()
            
            return analytics
            
        except (ValueError, IndexError) as e:
            return {"error": str(e)}
    
    def get_market_analytics(self, market_id: str) -> Optional[Dict]:
        """Get stored analytics for a market."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM market_analytics WHERE market_id = ?", (market_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def find_value_bets(self, min_edge: float = 0.05, min_volume: float = 10000,
                        limit: int = 20) -> List[Dict]:
        """
        Find markets with positive edge (potential value bets).
        
        Args:
            min_edge: Minimum edge required (default 5%)
            min_volume: Minimum market volume
            limit: Max results
        
        Returns:
            List of markets with positive edge, sorted by edge desc
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT ma.*, m.question, m.volume, m.category, m.end_date
            FROM market_analytics ma
            JOIN markets m ON ma.market_id = m.id
            WHERE ma.edge >= ? AND m.volume >= ? AND m.active = 1
            ORDER BY ma.edge DESC
            LIMIT ?
        """, (min_edge, min_volume, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(r) for r in rows]

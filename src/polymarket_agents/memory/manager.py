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
        cursor.execute(
            """
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
        """
        )

        # Add missing columns to existing tables (for schema evolution)
        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN category TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN outcome_prices TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN volume REAL")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN liquidity REAL")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN active BOOLEAN")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN end_date TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN slug TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN clob_token_ids TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN event_id TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN tags TEXT")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN last_updated TEXT")
        except sqlite3.OperationalError:
            pass

        # Create indexes for efficient querying (only if columns exist)
        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON markets(category)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_active ON markets(active)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_volume ON markets(volume)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_end_date ON markets(end_date)")
        except sqlite3.OperationalError:
            pass

        try:
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_category_volume ON markets(category, volume DESC)")
        except sqlite3.OperationalError:
            pass

        # News table
        cursor.execute(
            """
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
        """
        )

        # Price history table - track price movements over time
        cursor.execute(
            """
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
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_price_market ON price_history(market_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_price_timestamp ON price_history(timestamp)"
        )

        # Bets tracking table - user positions and P&L
        cursor.execute(
            """
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
        """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bets_market ON bets(market_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_bets_status ON bets(status)")

        # Research table - store gathered information for markets
        cursor.execute(
            """
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
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_research_market ON research(market_id)"
        )

        # Market analytics table - computed stats for betting
        cursor.execute(
            """
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
        """
        )

        # Agent execution tracking table - track every agent run
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                query TEXT,
                status TEXT NOT NULL,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT,
                duration_ms INTEGER,
                current_node TEXT,
                completed_nodes TEXT,
                result TEXT,
                error TEXT,
                tokens_used INTEGER,
                langsmith_run_id TEXT
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_executions_started ON agent_executions(started_at DESC)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_agent_executions_status ON agent_executions(status)"
        )

        # Node execution tracking table - track individual node runs
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS node_executions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_execution_id INTEGER NOT NULL,
                node_name TEXT NOT NULL,
                node_type TEXT,
                started_at TEXT DEFAULT CURRENT_TIMESTAMP,
                completed_at TEXT,
                duration_ms INTEGER,
                status TEXT,
                input_data TEXT,
                output_data TEXT,
                error TEXT,
                FOREIGN KEY (agent_execution_id) REFERENCES agent_executions(id)
            )
        """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_node_executions_agent ON node_executions(agent_execution_id)"
        )

        # Agent performance metrics table - aggregate statistics
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_type TEXT NOT NULL,
                time_period TEXT NOT NULL,
                total_runs INTEGER DEFAULT 0,
                successful_runs INTEGER DEFAULT 0,
                failed_runs INTEGER DEFAULT 0,
                avg_duration_ms INTEGER,
                avg_tokens_used INTEGER,
                total_bets INTEGER DEFAULT 0,
                winning_bets INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                win_rate REAL,
                sharpe_ratio REAL,
                calculated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(agent_type, time_period)
            )
        """
        )

        conn.commit()
        conn.close()

    def add_market(
        self, market_data: Dict[str, Any], news_data: List[Dict[str, Any]] = None
    ):
        """Add or update a market and its associated news."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Parse/Format fields
        m_id = str(market_data.get("id"))
        question = market_data.get("question")
        description = market_data.get("description", "")
        category = market_data.get("category", "other")

        # Handle list/dict fields by dumping to JSON
        outcomes = market_data.get("outcomes", [])
        if isinstance(outcomes, str):
            outcomes = outcomes  # Already JSON string
        else:
            outcomes = json.dumps(outcomes)

        outcome_prices = market_data.get("outcome_prices") or market_data.get(
            "outcomePrices", []
        )
        if isinstance(outcome_prices, str):
            outcome_prices = outcome_prices
        else:
            outcome_prices = json.dumps(outcome_prices)

        # Numeric fields
        volume = float(market_data.get("volume", 0) or 0)
        liquidity = float(market_data.get("liquidity", 0) or 0)

        active = 1 if market_data.get("active") else 0
        end_date = (
            market_data.get("endDate")
            or market_data.get("end_date")
            or market_data.get("end")
        )
        slug = market_data.get("slug", "")
        clob_token_ids = market_data.get("clobTokenIds") or market_data.get(
            "clob_token_ids", ""
        )
        event_id = market_data.get("event_id", "")

        # Tags handling
        tags = market_data.get("tags", [])
        if isinstance(tags, list):
            tags = json.dumps(tags)

        last_updated = datetime.now().isoformat()

        # Upsert Market
        cursor.execute(
            """
            INSERT OR REPLACE INTO markets 
            (id, question, description, category, outcomes, outcome_prices, volume, liquidity, 
             active, end_date, slug, clob_token_ids, event_id, tags, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                m_id,
                question,
                description,
                category,
                outcomes,
                outcome_prices,
                volume,
                liquidity,
                active,
                end_date,
                slug,
                clob_token_ids,
                event_id,
                tags,
                last_updated,
            ),
        )

        # Insert News
        if news_data:
            for article in news_data:
                cursor.execute(
                    """
                    INSERT INTO news (market_id, title, description, url, published_at, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        m_id,
                        article.get("title"),
                        article.get("description"),
                        article.get("url"),
                        article.get("published_at") or article.get("publishedAt"),
                        last_updated,
                    ),
                )

        conn.commit()
        conn.close()

    def _parse_market_row(self, row: sqlite3.Row) -> Dict:
        """Parse a market row into a dictionary with proper types."""
        d = dict(row)

        # Parse JSON fields
        try:
            d["outcomes"] = json.loads(d["outcomes"]) if d.get("outcomes") else []
        except (json.JSONDecodeError, TypeError):
            d["outcomes"] = []

        try:
            d["outcome_prices"] = (
                json.loads(d["outcome_prices"]) if d.get("outcome_prices") else []
            )
        except (json.JSONDecodeError, TypeError):
            d["outcome_prices"] = []

        try:
            d["tags"] = json.loads(d["tags"]) if d.get("tags") else []
        except (json.JSONDecodeError, TypeError):
            d["tags"] = []

        d["active"] = bool(d.get("active"))
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

        cursor.execute(
            "SELECT * FROM markets ORDER BY last_updated DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._parse_market_row(r) for r in rows]

    def list_markets_by_category(
        self, category: str, limit: int = 20, active_only: bool = True
    ) -> List[Dict]:
        """List markets in a specific category."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if active_only:
            cursor.execute(
                "SELECT * FROM markets WHERE category = ? AND active = 1 ORDER BY volume DESC LIMIT ?",
                (category, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM markets WHERE category = ? ORDER BY volume DESC LIMIT ?",
                (category, limit),
            )

        rows = cursor.fetchall()
        conn.close()

        return [self._parse_market_row(r) for r in rows]

    def list_top_volume_markets(
        self, limit: int = 20, category: str = None
    ) -> List[Dict]:
        """List markets with highest volume."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if category:
            cursor.execute(
                "SELECT * FROM markets WHERE category = ? AND active = 1 ORDER BY volume DESC LIMIT ?",
                (category, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM markets WHERE active = 1 ORDER BY volume DESC LIMIT ?",
                (limit,),
            )

        rows = cursor.fetchall()
        conn.close()

        return [self._parse_market_row(r) for r in rows]

    def search_markets(self, query: str, limit: int = 20, category: str = None) -> List[Dict]:
        """Search markets by question text, optionally filtered by category."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if category:
            cursor.execute(
                "SELECT * FROM markets WHERE question LIKE ? AND category = ? ORDER BY volume DESC LIMIT ?",
                (f"%{query}%", category, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM markets WHERE question LIKE ? ORDER BY volume DESC LIMIT ?",
                (f"%{query}%", limit),
            )

        rows = cursor.fetchall()
        conn.close()

        return [self._parse_market_row(r) for r in rows]

    def get_categories(self) -> List[Dict]:
        """Get all categories with market counts and total volume."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT category, COUNT(*) as count, SUM(volume) as total_volume
            FROM markets 
            WHERE active = 1
            GROUP BY category
            ORDER BY count DESC
        """
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {"category": r[0], "count": r[1], "total_volume": r[2] or 0} for r in rows
        ]

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
            "total_categories": total_categories,
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
        cursor.execute(
            """
            SELECT COUNT(*), SUM(volume)
            FROM markets 
            WHERE end_date IS NOT NULL 
            AND end_date != ''
            AND datetime(end_date) < datetime('now', ?)
        """,
            (f"-{grace_hours} hours",),
        )

        result = cursor.fetchone()
        expired_count = result[0] or 0
        expired_volume = result[1] or 0

        # Get category breakdown of expired markets
        cursor.execute(
            """
            SELECT category, COUNT(*) 
            FROM markets 
            WHERE end_date IS NOT NULL 
            AND end_date != ''
            AND datetime(end_date) < datetime('now', ?)
            GROUP BY category
        """,
            (f"-{grace_hours} hours",),
        )

        expired_by_category = {row[0]: row[1] for row in cursor.fetchall()}

        # Delete expired markets
        cursor.execute(
            """
            DELETE FROM markets 
            WHERE end_date IS NOT NULL 
            AND end_date != ''
            AND datetime(end_date) < datetime('now', ?)
        """,
            (f"-{grace_hours} hours",),
        )

        deleted_count = cursor.rowcount

        # Also delete associated news for expired markets
        cursor.execute(
            """
            DELETE FROM news 
            WHERE market_id NOT IN (SELECT id FROM markets)
        """
        )
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
            "grace_hours": grace_hours,
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

        cursor.execute(
            """
            SELECT * FROM markets 
            WHERE active = 1 
            AND end_date IS NOT NULL 
            AND end_date != ''
            AND datetime(end_date) > datetime('now')
            AND datetime(end_date) < datetime('now', ?)
            ORDER BY end_date ASC
            LIMIT ?
        """,
            (f"+{hours} hours", limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [self._parse_market_row(r) for r in rows]

    # =========================================================================
    # PRICE HISTORY METHODS
    # =========================================================================

    def record_price(
        self,
        market_id: str,
        yes_price: float,
        no_price: float = None,
        volume: float = None,
        liquidity: float = None,
    ):
        """Record a price snapshot for a market."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if no_price is None:
            no_price = 1.0 - yes_price

        cursor.execute(
            """
            INSERT INTO price_history (market_id, yes_price, no_price, volume, liquidity, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                market_id,
                yes_price,
                no_price,
                volume,
                liquidity,
                datetime.now().isoformat(),
            ),
        )

        conn.commit()
        conn.close()

    def get_price_history(self, market_id: str, hours: int = 24) -> List[Dict]:
        """Get price history for a market."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM price_history 
            WHERE market_id = ? 
            AND datetime(timestamp) > datetime('now', ?)
            ORDER BY timestamp ASC
        """,
            (market_id, f"-{hours} hours"),
        )

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
            "start_price": first["yes_price"],
            "end_price": last["yes_price"],
            "change": last["yes_price"] - first["yes_price"],
            "change_pct": (
                ((last["yes_price"] - first["yes_price"]) / first["yes_price"] * 100)
                if first["yes_price"] > 0
                else 0
            ),
            "data_points": len(history),
        }

    # =========================================================================
    # BETS TRACKING METHODS
    # =========================================================================

    def add_bet(
        self,
        market_id: str,
        side: str,
        entry_price: float,
        shares: float,
        strategy: str = None,
        confidence: float = None,
        notes: str = None,
    ) -> int:
        """Record a new bet/position."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get market question for reference
        market = self.get_market(market_id)
        question = market["question"] if market else "Unknown"

        cost_basis = entry_price * shares

        cursor.execute(
            """
            INSERT INTO bets (market_id, market_question, side, entry_price, current_price, 
                            shares, cost_basis, current_value, unrealized_pnl, status,
                            entry_date, strategy, confidence, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?, ?, ?)
        """,
            (
                market_id,
                question,
                side.upper(),
                entry_price,
                entry_price,
                shares,
                cost_basis,
                cost_basis,
                0.0,
                datetime.now().isoformat(),
                strategy,
                confidence,
                notes,
            ),
        )

        bet_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return bet_id

    def update_bet_prices(self):
        """Update all open bets with current market prices."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all open bets
        cursor.execute(
            "SELECT id, market_id, side, entry_price, shares FROM bets WHERE status = 'OPEN'"
        )
        open_bets = cursor.fetchall()

        for bet in open_bets:
            bet_id, market_id, side, entry_price, shares = bet
            market = self.get_market(market_id)

            if market and market.get("outcome_prices"):
                prices = market["outcome_prices"]
                if len(prices) >= 2:
                    try:
                        current_price = (
                            float(prices[0]) if side == "YES" else float(prices[1])
                        )
                        current_value = current_price * shares
                        unrealized_pnl = current_value - (entry_price * shares)

                        cursor.execute(
                            """
                            UPDATE bets SET current_price = ?, current_value = ?, unrealized_pnl = ?
                            WHERE id = ?
                        """,
                            (current_price, current_value, unrealized_pnl, bet_id),
                        )
                    except (ValueError, IndexError):
                        pass

        conn.commit()
        conn.close()

    def close_bet(self, bet_id: int, exit_price: float, status: str = "CLOSED"):
        """Close a bet with final price."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT shares, entry_price FROM bets WHERE id = ?", (bet_id,))
        row = cursor.fetchone()

        if row:
            shares, entry_price = row
            realized_pnl = (exit_price - entry_price) * shares

            cursor.execute(
                """
                UPDATE bets SET exit_price = ?, exit_date = ?, status = ?, 
                              realized_pnl = ?, current_price = ?, 
                              current_value = ?, unrealized_pnl = 0
                WHERE id = ?
            """,
                (
                    exit_price,
                    datetime.now().isoformat(),
                    status,
                    realized_pnl,
                    exit_price,
                    exit_price * shares,
                    bet_id,
                ),
            )

        conn.commit()
        conn.close()

    def get_open_bets(self) -> List[Dict]:
        """Get all open bets."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM bets WHERE status = 'OPEN' ORDER BY entry_date DESC"
        )
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
        cursor.execute(
            """
            SELECT COUNT(*), SUM(cost_basis), SUM(current_value), SUM(unrealized_pnl)
            FROM bets WHERE status = 'OPEN'
        """
        )
        open_row = cursor.fetchone()

        # Closed positions
        cursor.execute(
            """
            SELECT COUNT(*), SUM(realized_pnl) FROM bets WHERE status != 'OPEN'
        """
        )
        closed_row = cursor.fetchone()

        conn.close()

        return {
            "open_positions": open_row[0] or 0,
            "total_invested": open_row[1] or 0,
            "current_value": open_row[2] or 0,
            "unrealized_pnl": open_row[3] or 0,
            "closed_positions": closed_row[0] or 0,
            "realized_pnl": closed_row[1] or 0,
            "total_pnl": (open_row[3] or 0) + (closed_row[1] or 0),
        }

    # =========================================================================
    # RESEARCH METHODS
    # =========================================================================

    def add_research(
        self,
        market_id: str,
        research_type: str,
        content: str,
        source: str = None,
        sentiment_score: float = None,
        confidence: float = None,
        expires_hours: int = 24,
    ) -> int:
        """Add research/intelligence for a market."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        expires_at = None
        if expires_hours:
            from datetime import timedelta

            expires_at = (datetime.now() + timedelta(hours=expires_hours)).isoformat()

        cursor.execute(
            """
            INSERT INTO research (market_id, research_type, source, content, 
                                sentiment_score, confidence, created_at, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                market_id,
                research_type,
                source,
                content,
                sentiment_score,
                confidence,
                datetime.now().isoformat(),
                expires_at,
            ),
        )

        research_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return research_id

    def get_market_research(
        self, market_id: str, include_expired: bool = False
    ) -> List[Dict]:
        """Get all research for a market."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if include_expired:
            cursor.execute(
                """
                SELECT * FROM research WHERE market_id = ? ORDER BY created_at DESC
            """,
                (market_id,),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM research WHERE market_id = ? 
                AND (expires_at IS NULL OR datetime(expires_at) > datetime('now'))
                ORDER BY created_at DESC
            """,
                (market_id,),
            )

        rows = cursor.fetchall()
        conn.close()

        return [dict(r) for r in rows]

    # =========================================================================
    # MARKET ANALYTICS METHODS
    # =========================================================================

    def update_market_analytics(
        self, market_id: str, estimated_prob: float = None, analyst_notes: str = None
    ) -> Dict:
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

        prices = market.get("outcome_prices", [])
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
                "analyst_notes": analyst_notes,
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

            cursor.execute(
                """
                INSERT OR REPLACE INTO market_analytics 
                (market_id, implied_prob_yes, estimated_prob_yes, edge, expected_value,
                 kelly_fraction, volatility, price_momentum, last_analysis, analyst_notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    market_id,
                    analytics["implied_prob_yes"],
                    analytics["estimated_prob_yes"],
                    analytics["edge"],
                    analytics["expected_value"],
                    analytics["kelly_fraction"],
                    analytics["volatility"],
                    analytics["price_momentum"],
                    analytics["last_analysis"],
                    analytics["analyst_notes"],
                ),
            )

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

        cursor.execute(
            "SELECT * FROM market_analytics WHERE market_id = ?", (market_id,)
        )
        row = cursor.fetchone()
        conn.close()

        return dict(row) if row else None

    def find_value_bets(
        self, min_edge: float = 0.05, min_volume: float = 10000, limit: int = 20
    ) -> List[Dict]:
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

        cursor.execute(
            """
            SELECT ma.*, m.question, m.volume, m.category, m.end_date
            FROM market_analytics ma
            JOIN markets m ON ma.market_id = m.id
            WHERE ma.edge >= ? AND m.volume >= ? AND m.active = 1
            ORDER BY ma.edge DESC
            LIMIT ?
        """,
            (min_edge, min_volume, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(r) for r in rows]

    def get_markets_for_news_update(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get markets that need news updates (no recent news or high volume)."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT m.*, COUNT(n.id) as news_count,
                   MAX(n.published_at) as latest_news
            FROM markets m
            LEFT JOIN news n ON m.id = n.market_id
            WHERE m.active = 1 AND m.volume > 10000
            GROUP BY m.id
            HAVING news_count = 0 OR latest_news < datetime('now', '-1 day')
            ORDER BY m.volume DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        markets = []

        for row in rows:
            market = self._row_to_market_dict(row)
            market["news_count"] = row[-2]
            market["latest_news"] = row[-1]
            markets.append(market)

        conn.close()
        return markets

    def add_market_news(self, market_id: str, articles: List[Dict[str, Any]]) -> int:
        """Add news articles for a market."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        added_count = 0
        for article in articles:
            try:
                cursor.execute(
                    """
                    INSERT INTO news (market_id, title, description, url, published_at, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        market_id,
                        article.get("title", ""),
                        article.get("description", ""),
                        article.get("url", ""),
                        article.get("publishedAt", ""),
                        datetime.now().isoformat(),
                    ),
                )
                added_count += 1
            except sqlite3.IntegrityError:
                # Article already exists, skip
                continue

        conn.commit()
        conn.close()
        return added_count

    def cleanup_old_price_history(self, cutoff_date: datetime) -> int:
        """Remove old price history entries beyond retention period."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            DELETE FROM price_history
            WHERE datetime(timestamp) < ?
        """,
            (cutoff_date.isoformat(),),
        )

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return deleted_count

    def vacuum_database(self) -> float:
        """Vacuum database to reclaim space and return MB freed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get size before vacuum
        cursor.execute("PRAGMA page_count")
        pages_before = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        size_before = pages_before * page_size

        # Vacuum
        conn.execute("VACUUM")

        # Get size after vacuum
        cursor.execute("PRAGMA page_count")
        pages_after = cursor.fetchone()[0]
        size_after = pages_after * page_size

        conn.close()

        space_freed_mb = (size_before - size_after) / (1024 * 1024)
        return max(0, space_freed_mb)

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Basic counts
        cursor.execute("SELECT COUNT(*) FROM markets WHERE active = 1")
        stats["active_markets"] = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM markets")
        stats["total_markets"] = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(volume) FROM markets WHERE active = 1")
        stats["total_volume"] = cursor.fetchone()[0] or 0

        cursor.execute("SELECT SUM(liquidity) FROM markets WHERE active = 1")
        stats["total_liquidity"] = cursor.fetchone()[0] or 0

        # Category breakdown
        cursor.execute(
            """
            SELECT category, COUNT(*), SUM(volume)
            FROM markets
            WHERE active = 1
            GROUP BY category
            ORDER BY COUNT(*) DESC
        """
        )

        stats["markets_by_category"] = {}
        for row in cursor.fetchall():
            stats["markets_by_category"][row[0]] = {
                "count": row[1],
                "volume": row[2] or 0,
            }

        # Database size
        cursor.execute("PRAGMA page_count")
        pages = cursor.fetchone()[0]
        cursor.execute("PRAGMA page_size")
        page_size = cursor.fetchone()[0]
        stats["database_size_mb"] = (pages * page_size) / (1024 * 1024)

        # Last refresh
        cursor.execute("SELECT MAX(last_updated) FROM markets")
        last_update = cursor.fetchone()[0]
        stats["last_refresh"] = last_update

        conn.close()
        return stats

    # =========================================================================
    # AGENT EXECUTION TRACKING METHODS
    # =========================================================================

    def start_agent_execution(
        self, agent_type: str, agent_name: str, query: str = None
    ) -> int:
        """
        Start tracking a new agent execution.

        Args:
            agent_type: Type of agent ('memory_agent', 'planning_agent', 'subagent')
            agent_name: Specific agent identifier
            query: User query/input

        Returns:
            execution_id: ID of the new execution record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO agent_executions (agent_type, agent_name, query, status, started_at, completed_nodes)
            VALUES (?, ?, ?, 'running', ?, '[]')
        """,
            (agent_type, agent_name, query, datetime.now().isoformat()),
        )

        execution_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return execution_id

    def complete_agent_execution(
        self,
        execution_id: int,
        result: str = None,
        tokens_used: int = None,
        langsmith_run_id: str = None,
    ):
        """Complete an agent execution with results."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get start time to calculate duration
        cursor.execute(
            "SELECT started_at FROM agent_executions WHERE id = ?", (execution_id,)
        )
        row = cursor.fetchone()
        if row:
            started_at = datetime.fromisoformat(row[0])
            completed_at = datetime.now()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            cursor.execute(
                """
                UPDATE agent_executions
                SET status = 'completed', completed_at = ?, duration_ms = ?,
                    result = ?, tokens_used = ?, langsmith_run_id = ?
                WHERE id = ?
            """,
                (
                    completed_at.isoformat(),
                    duration_ms,
                    result,
                    tokens_used,
                    langsmith_run_id,
                    execution_id,
                ),
            )

        conn.commit()
        conn.close()

    def fail_agent_execution(self, execution_id: int, error: str):
        """Mark an agent execution as failed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get start time to calculate duration
        cursor.execute(
            "SELECT started_at FROM agent_executions WHERE id = ?", (execution_id,)
        )
        row = cursor.fetchone()
        if row:
            started_at = datetime.fromisoformat(row[0])
            completed_at = datetime.now()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            cursor.execute(
                """
                UPDATE agent_executions
                SET status = 'failed', completed_at = ?, duration_ms = ?, error = ?
                WHERE id = ?
            """,
                (completed_at.isoformat(), duration_ms, error, execution_id),
            )

        conn.commit()
        conn.close()

    def update_current_node(self, execution_id: int, node_name: str):
        """Update the currently executing node for an agent run."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get current completed nodes
        cursor.execute(
            "SELECT completed_nodes FROM agent_executions WHERE id = ?", (execution_id,)
        )
        row = cursor.fetchone()
        if row:
            completed_nodes = json.loads(row[0] or "[]")
            if node_name not in completed_nodes:
                # Mark previous node as completed
                cursor.execute(
                    "SELECT current_node FROM agent_executions WHERE id = ?",
                    (execution_id,),
                )
                prev_node = cursor.fetchone()
                if prev_node and prev_node[0]:
                    completed_nodes.append(prev_node[0])

            cursor.execute(
                """
                UPDATE agent_executions
                SET current_node = ?, completed_nodes = ?
                WHERE id = ?
            """,
                (node_name, json.dumps(completed_nodes), execution_id),
            )

        conn.commit()
        conn.close()

    def track_node_execution(
        self,
        agent_execution_id: int,
        node_name: str,
        node_type: str = None,
        input_data: str = None,
        output_data: str = None,
        error: str = None,
        duration_ms: int = None,
        status: str = "completed",
    ) -> int:
        """
        Track an individual node execution within an agent run.

        Args:
            agent_execution_id: Parent agent execution ID
            node_name: Name of the node
            node_type: Type of node ('retriever', 'tool', 'llm', 'decision')
            input_data: JSON snapshot of input state
            output_data: JSON snapshot of output state
            error: Error message if failed
            duration_ms: Execution duration
            status: 'completed', 'failed', or 'skipped'

        Returns:
            node_execution_id: ID of the node execution record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        started_at = datetime.now()
        completed_at = started_at if duration_ms else None

        cursor.execute(
            """
            INSERT INTO node_executions
            (agent_execution_id, node_name, node_type, started_at, completed_at,
             duration_ms, status, input_data, output_data, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                agent_execution_id,
                node_name,
                node_type,
                started_at.isoformat(),
                completed_at.isoformat() if completed_at else None,
                duration_ms,
                status,
                input_data,
                output_data,
                error,
            ),
        )

        node_execution_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return node_execution_id

    def get_recent_executions(self, limit: int = 50) -> List[Dict]:
        """Get recent agent executions."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM agent_executions
            ORDER BY started_at DESC
            LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [dict(r) for r in rows]

    def get_execution_metrics(self, time_period: str = "24h") -> Optional[Dict]:
        """
        Get execution metrics for a time period.

        Args:
            time_period: '1h', '24h', '7d', '30d'

        Returns:
            Metrics dict or None if no data
        """
        # Parse time period
        hours_map = {"1h": 1, "24h": 24, "7d": 168, "30d": 720}
        hours = hours_map.get(time_period, 24)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get execution stats
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_runs,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_runs,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_runs,
                AVG(duration_ms) as avg_duration_ms,
                AVG(tokens_used) as avg_tokens_used
            FROM agent_executions
            WHERE datetime(started_at) > datetime('now', ?)
        """,
            (f"-{hours} hours",),
        )

        row = cursor.fetchone()

        if not row or row[0] == 0:
            conn.close()
            return None

        # Get betting stats (from bets table) for the same period
        cursor.execute(
            """
            SELECT
                COUNT(*) as total_bets,
                SUM(CASE WHEN status = 'WON' THEN 1 ELSE 0 END) as winning_bets,
                SUM(realized_pnl) + SUM(unrealized_pnl) as total_pnl
            FROM bets
            WHERE datetime(entry_date) > datetime('now', ?)
        """,
            (f"-{hours} hours",),
        )

        bet_row = cursor.fetchone()
        conn.close()

        total_bets = bet_row[0] if bet_row else 0
        winning_bets = bet_row[1] if bet_row else 0
        total_pnl = bet_row[2] if bet_row else 0.0

        win_rate = winning_bets / total_bets if total_bets > 0 else 0.0

        return {
            "total_runs": row[0],
            "successful_runs": row[1],
            "failed_runs": row[2],
            "avg_duration_ms": int(row[3]) if row[3] else 0,
            "avg_tokens_used": int(row[4]) if row[4] else 0,
            "total_bets": total_bets,
            "winning_bets": winning_bets,
            "total_pnl": total_pnl,
            "win_rate": win_rate,
            "sharpe_ratio": 0.0,  # TODO: Calculate Sharpe ratio
        }

    def _row_to_market_dict(self, row) -> Dict[str, Any]:
        """Convert a database row to a market dictionary."""
        if isinstance(row, sqlite3.Row):
            return dict(row)
        elif (
            isinstance(row, tuple) and len(row) >= 14
        ):  # Based on the markets table schema
            return {
                "id": row[0],
                "question": row[1],
                "description": row[2],
                "category": row[3],
                "outcomes": row[4],
                "outcome_prices": row[5],
                "volume": row[6],
                "liquidity": row[7],
                "active": row[8],
                "end_date": row[9],
                "slug": row[10],
                "clob_token_ids": row[11],
                "event_id": row[12],
                "tags": row[13],
                "last_updated": row[14] if len(row) > 14 else None,
            }
        else:
            return {}

    def _row_to_news_dict(self, row) -> Dict[str, Any]:
        """Convert a database row to a news dictionary."""
        if isinstance(row, sqlite3.Row):
            return dict(row)
        elif isinstance(row, tuple) and len(row) >= 6:  # Based on the news table schema
            return {
                "id": row[0],
                "market_id": row[1],
                "title": row[2],
                "description": row[3],
                "url": row[4],
                "published_at": row[5],
                "ingested_at": row[6] if len(row) > 6 else None,
            }
        else:
            return {}

    def _row_to_price_dict(self, row) -> Dict[str, Any]:
        """Convert a database row to a price history dictionary."""
        if isinstance(row, sqlite3.Row):
            return dict(row)
        elif (
            isinstance(row, tuple) and len(row) >= 6
        ):  # Based on the price_history table schema
            return {
                "id": row[0],
                "market_id": row[1],
                "yes_price": row[2],
                "no_price": row[3],
                "volume": row[4],
                "liquidity": row[5],
                "timestamp": row[6],
            }
        else:
            return {}

    def _row_to_bet_dict(self, row) -> Dict[str, Any]:
        """Convert a database row to a bet dictionary."""
        if isinstance(row, sqlite3.Row):
            return dict(row)
        elif (
            isinstance(row, tuple) and len(row) >= 16
        ):  # Based on the bets table schema
            return {
                "id": row[0],
                "market_id": row[1],
                "market_question": row[2],
                "side": row[3],
                "entry_price": row[4],
                "current_price": row[5],
                "shares": row[6],
                "cost_basis": row[7],
                "current_value": row[8],
                "unrealized_pnl": row[9],
                "realized_pnl": row[10],
                "status": row[11],
                "entry_date": row[12],
                "exit_date": row[13],
                "exit_price": row[14],
                "notes": row[15],
            }
        else:
            return {}

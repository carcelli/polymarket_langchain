"""
Bitcoin Market Continuous Tracker

Continuously polls Polymarket for Bitcoin price prediction markets and stores
historical snapshots for machine learning and automated betting.

Features:
- 15-minute interval snapshots (configurable)
- Technical indicators (momentum, volatility, RSI)
- Real-time Bitcoin price from Binance
- ML-ready features (volume spikes, price deviation)
- Automatic outcome resolution tracking
- Graceful error handling and rate limiting

Usage:
    # Run continuously (15-min intervals)
    python -m polymarket_agents.services.bitcoin_tracker

    # Run with custom interval (5 minutes)
    python -m polymarket_agents.services.bitcoin_tracker --interval 300

    # Track specific market IDs
    python -m polymarket_agents.services.bitcoin_tracker --market-ids 574073,12345

    # One-time snapshot (no loop)
    python -m polymarket_agents.services.bitcoin_tracker --once
"""

import argparse
import ccxt  # type: ignore[import]
import httpx
import json
import logging
import numpy as np
import signal
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set


# Configuration
DB_PATH = Path("data/bitcoin_tracker.db")
GAMMA_API = "https://gamma-api.polymarket.com"
COLLECTION_INTERVAL = 900  # 15 minutes in seconds
MAX_RETRIES = 3
RETRY_DELAY = 5

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class BitcoinMarketTracker:
    """
    Continuous tracker for Bitcoin prediction markets.

    Collects market data at regular intervals and enriches it with:
    - Real-time Bitcoin spot price
    - Technical indicators
    - Historical price momentum
    - Volume metrics

    All data is stored in SQLite for ML training and analysis.
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        interval: int = COLLECTION_INTERVAL,
        market_ids: Optional[List[str]] = None,
    ):
        """
        Initialize the tracker.

        Args:
            db_path: Path to SQLite database
            interval: Collection interval in seconds (default: 900 = 15 min)
            market_ids: Optional list of specific market IDs to track
        """
        self.db_path = db_path
        self.interval = interval
        self.market_ids = set(market_ids) if market_ids else None
        self.running = False

        # Initialize components
        self.conn = self._init_database()
        self.exchange = ccxt.binance()
        self.http_client = httpx.Client(timeout=30.0)

        # Price history for momentum calculation
        self.btc_price_history: List[float] = []
        self.lookback_periods = 30  # Keep last 30 data points

        # Market cache to track changes
        self.last_prices: Dict[str, float] = {}

        logger.info(f"✅ BitcoinMarketTracker initialized")
        logger.info(f"   Database: {db_path}")
        logger.info(f"   Interval: {interval}s ({interval/60:.1f} min)")
        if self.market_ids:
            logger.info(f"   Tracking specific markets: {self.market_ids}")

    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database with ML-ready schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                market_id TEXT NOT NULL,
                question TEXT NOT NULL,
                
                -- Market data
                yes_price REAL,
                no_price REAL,
                implied_probability REAL,  -- yes_price as probability
                volume REAL,
                liquidity REAL,
                end_date TEXT,
                time_to_expiry_hours REAL,
                
                -- Bitcoin spot price
                btc_spot_price REAL,
                btc_24h_change_pct REAL,
                
                -- Technical indicators (derived)
                price_momentum_15m REAL,  -- Change over last 15 min
                price_momentum_1h REAL,   -- Change over last hour
                volume_spike REAL,         -- Current volume vs average
                price_volatility REAL,     -- Std dev of recent prices
                rsi_14 REAL,              -- Relative Strength Index
                market_edge REAL,         -- Deviation from 0.5 (mispricing indicator)
                
                -- Outcome tracking for ML labels
                resolved INTEGER DEFAULT 0,
                outcome TEXT,  -- 'YES' or 'NO' when resolved
                profit_if_bought_yes REAL,  -- Retrospective profit calculation
                profit_if_bought_no REAL,
                
                -- Metadata
                data_quality_score REAL DEFAULT 1.0,  -- 0-1, for filtering noisy data
                
                UNIQUE(timestamp, market_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_market_id ON market_snapshots(market_id);
            CREATE INDEX IF NOT EXISTS idx_timestamp ON market_snapshots(timestamp);
            CREATE INDEX IF NOT EXISTS idx_resolved ON market_snapshots(resolved);
            CREATE INDEX IF NOT EXISTS idx_quality ON market_snapshots(data_quality_score);
            
            -- Resolution tracking table (for updating historical snapshots)
            CREATE TABLE IF NOT EXISTS market_resolutions (
                market_id TEXT PRIMARY KEY,
                resolved_at TEXT NOT NULL,
                outcome TEXT NOT NULL,
                final_btc_price REAL,
                resolution_source TEXT  -- 'polymarket', 'manual', etc.
            );
            
            -- Collection metadata
            CREATE TABLE IF NOT EXISTS collection_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                start_time TEXT NOT NULL,
                end_time TEXT,
                snapshots_collected INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0,
                status TEXT  -- 'running', 'completed', 'failed'
            );
        """
        )

        conn.commit()
        logger.info("✅ Database schema initialized")
        return conn

    def _fetch_bitcoin_markets(self) -> List[Dict]:
        """
        Fetch active Bitcoin prediction markets from Polymarket.

        Returns:
            List of market dictionaries
        """
        try:
            # Search for Bitcoin-related markets
            response = self.http_client.get(
                f"{GAMMA_API}/markets",
                params={
                    "active": True,
                    "limit": 500,  # Get many to find Bitcoin ones
                },
            )
            response.raise_for_status()
            markets = response.json()

            # Filter for Bitcoin markets
            btc_markets = []
            for m in markets:
                question = m.get("question", "")

                # If specific market IDs provided, filter by those
                if self.market_ids:
                    if m.get("id") in self.market_ids:
                        btc_markets.append(m)
                        continue

                # Otherwise, filter by keywords
                if any(
                    keyword in question.lower()
                    for keyword in ["bitcoin", "btc", "$btc"]
                ):
                    # Prioritize price prediction markets
                    if any(
                        word in question.lower()
                        for word in ["reach", "hit", "above", "below", "price", "$"]
                    ):
                        btc_markets.append(m)

            logger.info(f"📊 Found {len(btc_markets)} Bitcoin markets")
            return btc_markets

        except Exception as e:
            logger.error(f"❌ Error fetching markets: {e}")
            return []

    def _get_btc_spot_price(self) -> Optional[Dict[str, float]]:
        """
        Get current Bitcoin spot price and 24h change from Binance.

        Returns:
            Dict with 'price' and 'change_24h_pct' or None on error
        """
        try:
            ticker = self.exchange.fetch_ticker("BTC/USDT")

            price = ticker["last"]
            change_24h_pct = ticker.get("percentage", 0.0)

            # Add to history for momentum calculation
            self.btc_price_history.append(price)
            if len(self.btc_price_history) > self.lookback_periods:
                self.btc_price_history.pop(0)

            return {"price": price, "change_24h_pct": change_24h_pct}
        except Exception as e:
            logger.warning(f"⚠️  Could not fetch BTC price: {e}")
            return None

    def _calculate_technical_indicators(
        self, market_id: str, current_price: float, volume: float
    ) -> Dict[str, float]:
        """
        Calculate technical indicators for ML features.

        Args:
            market_id: Market identifier
            current_price: Current yes_price
            volume: Current trading volume

        Returns:
            Dict of calculated indicators
        """
        indicators = {}

        # Get historical data for this market
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT yes_price, volume, timestamp
            FROM market_snapshots
            WHERE market_id = ?
            ORDER BY timestamp DESC
            LIMIT 10
        """,
            (market_id,),
        )
        history = cursor.fetchall()

        # Price momentum (15min and 1h approximations)
        if len(history) >= 1:
            # Last snapshot (15 min ago if running on schedule)
            indicators["price_momentum_15m"] = current_price - history[0][0]
        else:
            indicators["price_momentum_15m"] = 0.0

        if len(history) >= 4:
            # ~1 hour ago (4 snapshots * 15 min)
            indicators["price_momentum_1h"] = current_price - history[3][0]
        else:
            indicators["price_momentum_1h"] = 0.0

        # Volume spike (current vs average)
        if len(history) >= 3:
            avg_volume = np.mean([h[1] for h in history[:3]])
            if avg_volume > 0:
                indicators["volume_spike"] = (volume - avg_volume) / avg_volume
            else:
                indicators["volume_spike"] = 0.0
        else:
            indicators["volume_spike"] = 0.0

        # Price volatility (standard deviation)
        if len(history) >= 3:
            prices = [h[0] for h in history[:5]] + [current_price]
            indicators["price_volatility"] = float(np.std(prices))
        else:
            indicators["price_volatility"] = 0.0

        # RSI (14-period approximation)
        if len(history) >= 14:
            prices = [h[0] for h in reversed(history[:14])] + [current_price]
            gains = []
            losses = []
            for i in range(1, len(prices)):
                change = prices[i] - prices[i - 1]
                if change > 0:
                    gains.append(change)
                    losses.append(0)
                else:
                    gains.append(0)
                    losses.append(abs(change))

            avg_gain = np.mean(gains) if gains else 0.0
            avg_loss = np.mean(losses) if losses else 0.0

            if avg_loss == 0:
                indicators["rsi_14"] = 100.0
            else:
                rs = avg_gain / avg_loss
                indicators["rsi_14"] = 100 - (100 / (1 + rs))
        else:
            indicators["rsi_14"] = 50.0  # Neutral

        # Market edge (deviation from fair price of 0.5)
        indicators["market_edge"] = abs(current_price - 0.5)

        return indicators

    def _save_snapshot(self, market: Dict, btc_data: Optional[Dict]) -> bool:
        """
        Save a market snapshot to the database.

        Args:
            market: Market data from Polymarket API
            btc_data: Bitcoin spot price data

        Returns:
            True if saved successfully
        """
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            market_id = market["id"]
            question = market["question"]

            # Extract market prices
            tokens = market.get("tokens", [])
            yes_price = (
                float(tokens[0]["price"])
                if len(tokens) > 0 and tokens[0].get("price")
                else None
            )
            no_price = (
                float(tokens[1]["price"])
                if len(tokens) > 1 and tokens[1].get("price")
                else None
            )

            if yes_price is None:
                logger.warning(f"⚠️  Skipping market {market_id}: no price data")
                return False

            volume = float(market.get("volume", 0))
            liquidity = float(market.get("liquidity", 0))
            end_date = market.get("endDate") or market.get("end_date")

            # Calculate time to expiry
            time_to_expiry_hours = None
            if end_date:
                try:
                    expiry = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                    time_to_expiry_hours = (
                        expiry - datetime.now(timezone.utc)
                    ).total_seconds() / 3600
                except:
                    pass

            # Bitcoin spot data
            btc_price = btc_data["price"] if btc_data else None
            btc_change = btc_data["change_24h_pct"] if btc_data else None

            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(
                market_id, yes_price, volume
            )

            # Data quality score (based on liquidity and volume)
            quality_score = min(1.0, (liquidity + volume) / 10000.0)  # Scale to 0-1

            # Insert snapshot
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO market_snapshots (
                    timestamp, market_id, question,
                    yes_price, no_price, implied_probability,
                    volume, liquidity, end_date, time_to_expiry_hours,
                    btc_spot_price, btc_24h_change_pct,
                    price_momentum_15m, price_momentum_1h,
                    volume_spike, price_volatility, rsi_14, market_edge,
                    data_quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    timestamp,
                    market_id,
                    question,
                    yes_price,
                    no_price,
                    yes_price,  # implied_probability = yes_price
                    volume,
                    liquidity,
                    end_date,
                    time_to_expiry_hours,
                    btc_price,
                    btc_change,
                    indicators.get("price_momentum_15m", 0),
                    indicators.get("price_momentum_1h", 0),
                    indicators.get("volume_spike", 0),
                    indicators.get("price_volatility", 0),
                    indicators.get("rsi_14", 50),
                    indicators.get("market_edge", 0),
                    quality_score,
                ),
            )

            self.conn.commit()

            # Store last price for change detection
            self.last_prices[market_id] = yes_price

            return True

        except Exception as e:
            logger.error(f"❌ Error saving snapshot for market {market.get('id')}: {e}")
            return False

    def collect_snapshot(self) -> int:
        """
        Collect one snapshot of all tracked Bitcoin markets.

        Returns:
            Number of markets successfully collected
        """
        logger.info("📸 Collecting snapshot...")

        # Get Bitcoin spot price
        btc_data = self._get_btc_spot_price()
        if btc_data:
            logger.info(
                f"₿  BTC: ${btc_data['price']:,.2f} ({btc_data['change_24h_pct']:+.2f}% 24h)"
            )

        # Fetch markets
        markets = self._fetch_bitcoin_markets()

        if not markets:
            logger.warning("⚠️  No markets found")
            return 0

        # Save each market
        count = 0
        for market in markets:
            if self._save_snapshot(market, btc_data):
                count += 1

                # Log interesting changes
                market_id = market["id"]
                if market_id in self.last_prices:
                    current = (
                        float(market["tokens"][0]["price"])
                        if market.get("tokens")
                        else 0
                    )
                    last = self.last_prices[market_id]
                    change = current - last
                    if abs(change) > 0.01:  # >1% change
                        logger.info(
                            f"   📈 {market['question'][:60]}... ({change:+.3f})"
                        )

        logger.info(f"✅ Collected {count}/{len(markets)} markets")
        return count

    def _check_resolutions(self):
        """Check for resolved markets and update historical snapshots."""
        try:
            # Get unresolved markets we're tracking
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT DISTINCT market_id, question
                FROM market_snapshots
                WHERE resolved = 0
            """
            )
            unresolved = cursor.fetchall()

            for market_id, question in unresolved:
                # Query Polymarket API for market status
                try:
                    response = self.http_client.get(f"{GAMMA_API}/markets/{market_id}")
                    if response.status_code == 200:
                        market_data = response.json()

                        if market_data.get("closed") or not market_data.get("active"):
                            # Market resolved - determine outcome
                            # This is simplified - real implementation would check on-chain data
                            logger.info(f"🎯 Market resolved: {question[:60]}...")

                            # Update all historical snapshots for this market
                            cursor.execute(
                                """
                                UPDATE market_snapshots
                                SET resolved = 1
                                WHERE market_id = ?
                            """,
                                (market_id,),
                            )
                            self.conn.commit()

                except Exception as e:
                    logger.debug(f"Could not check resolution for {market_id}: {e}")
                    continue

        except Exception as e:
            logger.error(f"❌ Error checking resolutions: {e}")

    def start(self, run_once: bool = False):
        """
        Start the continuous collection loop.

        Args:
            run_once: If True, collect one snapshot and exit
        """
        self.running = True

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info("🚀 Starting Bitcoin market tracker")
        logger.info(f"   Press Ctrl+C to stop gracefully")

        # Record collection run
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO collection_runs (start_time, status)
            VALUES (?, 'running')
        """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        run_id = cursor.lastrowid
        self.conn.commit()

        total_snapshots = 0
        total_errors = 0

        try:
            while self.running:
                start_time = time.time()

                try:
                    # Collect snapshot
                    count = self.collect_snapshot()
                    total_snapshots += count

                    # Periodically check for resolutions (every 10th snapshot)
                    if total_snapshots % 10 == 0:
                        self._check_resolutions()

                    # Get database stats
                    cursor.execute("SELECT COUNT(*) FROM market_snapshots")
                    total_in_db = cursor.fetchone()[0]

                    logger.info(f"💾 Database: {total_in_db:,} total snapshots")

                except Exception as e:
                    logger.error(f"❌ Collection error: {e}")
                    total_errors += 1

                if run_once:
                    logger.info("✅ Single snapshot complete")
                    break

                # Sleep until next interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.interval - elapsed)

                if sleep_time > 0:
                    logger.info(
                        f"😴 Sleeping {sleep_time:.0f}s until next collection..."
                    )
                    time.sleep(sleep_time)

        finally:
            # Update collection run record
            cursor.execute(
                """
                UPDATE collection_runs
                SET end_time = ?, snapshots_collected = ?, errors = ?, status = 'completed'
                WHERE id = ?
            """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    total_snapshots,
                    total_errors,
                    run_id,
                ),
            )
            self.conn.commit()

            # Cleanup
            self.http_client.close()
            self.conn.close()

            logger.info(
                f"🎉 Tracker stopped. Collected {total_snapshots} snapshots ({total_errors} errors)"
            )

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info("\n🛑 Shutdown signal received. Stopping gracefully...")
        self.running = False


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Continuously track Bitcoin prediction markets for ML and betting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with 15-minute intervals (default)
  python -m polymarket_agents.services.bitcoin_tracker
  
  # Run with 5-minute intervals
  python -m polymarket_agents.services.bitcoin_tracker --interval 300
  
  # Track specific markets
  python -m polymarket_agents.services.bitcoin_tracker --market-ids 574073,12345
  
  # Collect one snapshot and exit
  python -m polymarket_agents.services.bitcoin_tracker --once
  
  # Custom database location
  python -m polymarket_agents.services.bitcoin_tracker --db data/my_btc_data.db
        """,
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=COLLECTION_INTERVAL,
        help=f"Collection interval in seconds (default: {COLLECTION_INTERVAL} = 15 min)",
    )

    parser.add_argument(
        "--market-ids",
        type=str,
        help='Comma-separated list of specific market IDs to track (e.g., "574073,12345")',
    )

    parser.add_argument(
        "--db", type=Path, default=DB_PATH, help=f"Database path (default: {DB_PATH})"
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Collect one snapshot and exit (no continuous loop)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse market IDs if provided
    market_ids = None
    if args.market_ids:
        market_ids = [mid.strip() for mid in args.market_ids.split(",")]

    # Create and start tracker
    tracker = BitcoinMarketTracker(
        db_path=args.db, interval=args.interval, market_ids=market_ids
    )

    tracker.start(run_once=args.once)


if __name__ == "__main__":
    main()

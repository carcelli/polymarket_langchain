"""
Crypto Market Data Collector

Collects 15-minute market data for ML training.
Stores snapshots with prices, indicators, and outcomes.

Usage:
    python -m polymarket_agents.domains.crypto.data_collector --interval 60
"""

import sqlite3
import time
import json
import ccxt  # type: ignore[import]
import httpx
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


DB_PATH = Path("data/crypto_ml.db")
GAMMA_API = "https://gamma-api.polymarket.com"
CRYPTO_TAG_ID = "21"


def init_database(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database for ML data."""

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS market_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            market_id TEXT NOT NULL,
            asset TEXT NOT NULL,
            question TEXT,
            yes_price REAL,
            no_price REAL,
            volume REAL,
            expiry_minutes REAL,
            end_date TEXT,

            -- Exchange data
            current_price REAL,

            -- Technical indicators
            momentum_5m REAL,
            momentum_30m REAL,
            volatility REAL,
            volume_spike REAL,
            rsi REAL,
            deviation REAL,

            -- For tracking resolution
            resolved INTEGER DEFAULT 0,
            outcome TEXT,  -- 'YES' or 'NO'

            UNIQUE(timestamp, market_id)
        );

        CREATE INDEX IF NOT EXISTS idx_market_id ON market_snapshots(market_id);
        CREATE INDEX IF NOT EXISTS idx_asset ON market_snapshots(asset);
        CREATE INDEX IF NOT EXISTS idx_resolved ON market_snapshots(resolved);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON market_snapshots(timestamp);
    """
    )

    conn.commit()
    return conn


class CryptoDataCollector:
    """Collects and stores crypto market data for ML."""

    def __init__(self, db_path: Path = DB_PATH):
        self.conn = init_database(db_path)
        self.exchange = ccxt.binance()
        self.lookback = 30

    def collect_snapshot(self) -> int:
        """
        Collect one snapshot of all active markets.

        Returns:
            Number of markets collected
        """
        markets = self._fetch_markets()

        if not markets:
            return 0

        count = 0
        timestamp = datetime.now(timezone.utc).isoformat()

        for market in markets:
            try:
                # Get exchange data and indicators
                data = self._enrich_market(market)
                if not data:
                    continue

                # Insert into database
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO market_snapshots (
                        timestamp, market_id, asset, question,
                        yes_price, no_price, volume, expiry_minutes, end_date,
                        current_price, momentum_5m, momentum_30m,
                        volatility, volume_spike, rsi, deviation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        timestamp,
                        market["id"],
                        market["asset"],
                        market["question"],
                        market["yes_price"],
                        market["no_price"],
                        market["volume"],
                        market["expiry_min"],
                        market["end_date"],
                        data["current_price"],
                        data["momentum_5m"],
                        data["momentum_30m"],
                        data["volatility"],
                        data["volume_spike"],
                        data["rsi"],
                        data["deviation"],
                    ),
                )

                count += 1

            except Exception as e:
                print(f"Error collecting {market.get('asset')}: {e}")
                continue

        self.conn.commit()
        return count

    def _fetch_markets(self, max_duration: int = 60) -> List[Dict]:
        """Fetch active crypto markets (Up/Down style)."""

        markets = []
        offset = 0
        limit = 50
        now = datetime.now(timezone.utc)

        while offset < 500:  # Paginate deeper to find Up/Down markets
            try:
                resp = httpx.get(
                    f"{GAMMA_API}/events",
                    params={
                        "limit": limit,
                        "offset": offset,
                        "closed": "false",
                        "tag_id": CRYPTO_TAG_ID,
                    },
                    timeout=10,
                )

                if resp.status_code != 200:
                    break

                events = resp.json()
                if not events:
                    break

                for event in events:
                    # Focus on "Up or Down" 15-minute markets
                    title = event.get("title", "").lower()
                    if "up or down" not in title:
                        continue

                    for market in event.get("markets", []):
                        end_str = market.get("endDate")
                        if not end_str:
                            continue

                        # Parse expiry
                        if end_str.endswith("Z"):
                            end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                        else:
                            end = datetime.fromisoformat(end_str)
                            if end.tzinfo is None:
                                end = end.replace(tzinfo=timezone.utc)

                        minutes_left = (end - now).total_seconds() / 60

                        if not (0 < minutes_left <= max_duration):
                            continue

                        # Parse prices
                        prices_raw = market.get("outcomePrices", "[]")
                        if isinstance(prices_raw, str):
                            prices = json.loads(prices_raw)
                        else:
                            prices = prices_raw

                        yes_price = float(prices[0]) if prices else 0.5
                        no_price = float(prices[1]) if len(prices) > 1 else 0.5

                        # Extract asset
                        question = market.get("question", "")
                        asset = self._extract_asset(question)

                        if asset:
                            markets.append(
                                {
                                    "id": market.get("id"),
                                    "asset": asset,
                                    "question": question,
                                    "yes_price": yes_price,
                                    "no_price": no_price,
                                    "volume": float(market.get("volume", 0)),
                                    "expiry_min": minutes_left,
                                    "end_date": end_str,
                                }
                            )

                offset += limit

            except Exception as e:
                print(f"Error fetching markets: {e}")
                break

        return markets

    def _extract_asset(self, question: str) -> Optional[str]:
        """Extract asset from question."""
        q = question.lower()

        if "bitcoin" in q:
            return "BTC"
        elif "ethereum" in q:
            return "ETH"
        elif "solana" in q:
            return "SOL"
        elif "xrp" in q:
            return "XRP"
        elif "doge" in q:
            return "DOGE"

        return None

    def _enrich_market(self, market: Dict) -> Optional[Dict]:
        """Add exchange data and indicators."""

        asset = market["asset"]
        symbol = f"{asset}/USDT"

        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, "1m", limit=self.lookback)

            if len(ohlcv) < self.lookback:
                return None

            closes = np.array([x[4] for x in ohlcv])
            volumes = np.array([x[5] for x in ohlcv])

            current_price = closes[-1]

            # Calculate indicators
            momentum_5m = (closes[-5:].mean() - closes[-10:-5].mean()) / closes[
                -10:-5
            ].mean()
            momentum_30m = (closes[-1] - closes[0]) / closes[0]
            volatility = closes.std() / closes.mean()

            recent_vol = volumes[-5:].mean()
            avg_vol = volumes.mean()
            volume_spike = (recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0

            # RSI
            deltas = np.diff(closes)
            gains = np.maximum(deltas, 0)
            losses = np.abs(np.minimum(deltas, 0))
            avg_gain = gains[-14:].mean()
            avg_loss = losses[-14:].mean()

            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

            # Deviation from mean
            mean_price = closes.mean()
            deviation = (closes[-1] - mean_price) / mean_price

            return {
                "current_price": current_price,
                "momentum_5m": momentum_5m,
                "momentum_30m": momentum_30m,
                "volatility": volatility,
                "volume_spike": volume_spike,
                "rsi": rsi,
                "deviation": deviation,
            }

        except Exception as e:
            print(f"Error enriching {asset}: {e}")
            return None

    def update_resolutions(self) -> int:
        """
        Check and update resolved markets.

        Returns:
            Number of markets updated
        """
        # Get unresolved markets that should have expired
        cursor = self.conn.execute(
            """
            SELECT DISTINCT market_id, end_date
            FROM market_snapshots
            WHERE resolved = 0
            AND datetime(end_date) < datetime('now')
        """
        )

        updated = 0

        for row in cursor.fetchall():
            market_id, end_date = row

            try:
                # Fetch market status from Gamma
                resp = httpx.get(f"{GAMMA_API}/markets/{market_id}", timeout=10)

                if resp.status_code != 200:
                    continue

                data = resp.json()

                # Check if resolved
                if data.get("closed"):
                    # Determine outcome from final prices
                    prices = data.get("outcomePrices", [])
                    if isinstance(prices, str):
                        prices = json.loads(prices)

                    if prices:
                        yes_final = float(prices[0])
                        outcome = "YES" if yes_final > 0.5 else "NO"

                        self.conn.execute(
                            """
                            UPDATE market_snapshots
                            SET resolved = 1, outcome = ?
                            WHERE market_id = ?
                        """,
                            (outcome, market_id),
                        )

                        updated += 1

            except Exception as e:
                print(f"Error checking resolution for {market_id}: {e}")
                continue

        self.conn.commit()
        return updated

    def get_stats(self) -> Dict:
        """Get collection statistics."""

        cursor = self.conn.execute(
            """
            SELECT
                COUNT(*) as total,
                COUNT(DISTINCT market_id) as unique_markets,
                SUM(CASE WHEN resolved = 1 THEN 1 ELSE 0 END) as resolved,
                MIN(timestamp) as first_snapshot,
                MAX(timestamp) as last_snapshot
            FROM market_snapshots
        """
        )

        row = cursor.fetchone()

        return {
            "total_snapshots": row[0] or 0,
            "unique_markets": row[1] or 0,
            "resolved_markets": row[2] or 0,
            "first_snapshot": row[3] or "N/A",
            "last_snapshot": row[4] or "N/A",
        }

    def export_training_data(self, output_path: str = "data/crypto_training.csv"):
        """Export resolved markets as training data."""

        import pandas as pd

        df = pd.read_sql_query(
            """
            SELECT
                asset,
                yes_price,
                no_price,
                volume,
                expiry_minutes,
                current_price,
                momentum_5m,
                momentum_30m,
                volatility,
                volume_spike,
                rsi,
                deviation,
                outcome
            FROM market_snapshots
            WHERE resolved = 1
            AND outcome IS NOT NULL
        """,
            self.conn,
        )

        # Create target variable
        df["target"] = (df["outcome"] == "YES").astype(int)

        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} samples to {output_path}")

        return df

    def close(self):
        """Close database connection."""
        self.conn.close()


def main():
    """Run data collection loop."""

    import argparse

    parser = argparse.ArgumentParser(description="Collect crypto market data for ML")
    parser.add_argument(
        "--interval", type=int, default=60, help="Collection interval in seconds"
    )
    parser.add_argument(
        "--duration", type=int, default=0, help="Total duration in minutes (0=infinite)"
    )
    parser.add_argument(
        "--update-only",
        action="store_true",
        help="Only update resolutions, don't collect",
    )
    parser.add_argument(
        "--export", action="store_true", help="Export training data and exit"
    )
    parser.add_argument("--stats", action="store_true", help="Show stats and exit")
    args = parser.parse_args()

    collector = CryptoDataCollector()

    # Stats only
    if args.stats:
        stats = collector.get_stats()
        print("\n📊 Collection Statistics:")
        print(f"   Total snapshots: {stats['total_snapshots']:,}")
        print(f"   Unique markets: {stats['unique_markets']:,}")
        print(f"   Resolved: {stats['resolved_markets']:,}")
        print(f"   First: {stats['first_snapshot']}")
        print(f"   Last: {stats['last_snapshot']}")
        collector.close()
        return

    # Export only
    if args.export:
        df = collector.export_training_data()
        print(f"\n✅ Exported {len(df)} training samples")
        collector.close()
        return

    # Update resolutions only
    if args.update_only:
        updated = collector.update_resolutions()
        print(f"✅ Updated {updated} market resolutions")
        collector.close()
        return

    # Collection loop
    print(f"\n🔄 Starting data collection (interval={args.interval}s)")
    print(f"   Database: {DB_PATH}")
    print("   Press Ctrl+C to stop\n")

    start_time = time.time()
    total_collected = 0

    try:
        while True:
            # Collect snapshot
            count = collector.collect_snapshot()
            total_collected += count

            # Update resolutions
            resolved = collector.update_resolutions()

            # Show progress
            stats = collector.get_stats()
            print(
                f"[{datetime.now().strftime('%H:%M:%S')}] "
                f"Collected: {count} | Total: {stats['total_snapshots']:,} | "
                f"Resolved: {stats['resolved_markets']:,} (+{resolved})"
            )

            # Check duration limit
            if args.duration > 0:
                elapsed = (time.time() - start_time) / 60
                if elapsed >= args.duration:
                    print(f"\n⏱️ Duration limit reached ({args.duration} min)")
                    break

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\n👋 Stopped by user")

    finally:
        stats = collector.get_stats()
        print("\n📊 Final Statistics:")
        print(f"   Total snapshots: {stats['total_snapshots']:,}")
        print(f"   Unique markets: {stats['unique_markets']:,}")
        print(f"   Resolved: {stats['resolved_markets']:,}")
        collector.close()


if __name__ == "__main__":
    main()

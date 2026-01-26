#!/usr/bin/env python3
"""
Production-grade paper trading simulator for Up/Down markets.

Isolates risk while validating edge: tracks live markets, places virtual bets,
resolves against actual outcomes, computes P&L over hundreds of markets.

This is event-driven, thread-safe, persistent — ready for production.
"""

import time
import sqlite3
import threading
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
import json

# DB schema: Persistent virtual portfolio across runs
DB_PATH = "data/simulator.db"


def init_db() -> sqlite3.Connection:
    """Initialize database with proper schema."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT UNIQUE,
            asset TEXT,
            duration_min INTEGER,
            start_time TEXT,
            end_time TEXT,
            start_price REAL,
            predicted_dir TEXT,      -- 'UP', 'DOWN', or NULL (no bet)
            confidence REAL,
            virtual_bet_usd REAL,    -- Fixed or Kelly-sized
            outcome_dir TEXT,        -- Resolved 'UP' or 'DOWN'
            actual_end_price REAL,
            virtual_profit REAL,     -- +profit or -loss
            resolved BOOLEAN DEFAULT FALSE,
            created_at TEXT,
            resolved_at TEXT
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS portfolio (
            key TEXT PRIMARY KEY,
            virtual_bankroll REAL,
            total_bets INTEGER DEFAULT 0,
            winning_bets INTEGER DEFAULT 0,
            total_profit REAL DEFAULT 0.0,
            max_bankroll REAL DEFAULT 1000.0,
            last_updated TEXT
        )
    """
    )

    # Seed initial bankroll if new
    conn.execute(
        """
        INSERT OR IGNORE INTO portfolio 
        (key, virtual_bankroll, max_bankroll, last_updated) 
        VALUES ('main', 1000.0, 1000.0, ?)
    """,
        (datetime.now(timezone.utc).isoformat(),),
    )

    conn.commit()
    return conn


class Simulator:
    """
    Production-grade paper trading simulator.

    Features:
    - Thread-safe database writes
    - Automatic market discovery
    - Outcome polling and resolution
    - Kelly criterion position sizing
    - Comprehensive performance tracking
    """

    def __init__(
        self,
        min_confidence: float = 0.60,
        risk_per_trade: float = 0.02,
        poll_interval: int = 30,
    ):
        """
        Args:
            min_confidence: Minimum prediction confidence to bet
            risk_per_trade: Fraction of bankroll to risk per bet (Kelly)
            poll_interval: Seconds between market checks
        """
        self.conn = init_db()
        self.lock = threading.Lock()
        self.min_confidence = min_confidence
        self.risk_per_trade = risk_per_trade
        self.poll_interval = poll_interval

        # Load current state
        with self.lock:
            row = self.conn.execute(
                "SELECT virtual_bankroll, total_bets, winning_bets, total_profit, max_bankroll "
                "FROM portfolio WHERE key='main'"
            ).fetchone()

            self.bankroll = row[0]
            self.total_bets = row[1]
            self.winning_bets = row[2]
            self.total_profit = row[3]
            self.max_bankroll = row[4]

        self.known_markets = set()

    def get_open_markets(self) -> List[Dict]:
        """
        Fetch active Up/Down markets from Gamma API.
        Returns list of markets with metadata.
        """
        url = "https://gamma-api.polymarket.com/markets"
        params = {"limit": 500, "closed": False}

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            markets = response.json()

            now = datetime.now(timezone.utc)
            updown_markets = []

            for market in markets:
                question = market.get("question", "")

                if "up or down" not in question.lower():
                    continue

                end_date_str = market.get("end_date_iso")
                if not end_date_str:
                    continue

                try:
                    end_date = datetime.fromisoformat(
                        end_date_str.replace("Z", "+00:00")
                    )
                    start_date_str = market.get("start_date_iso", "")

                    # Only markets ending in next 60 minutes
                    minutes_until_end = (end_date - now).total_seconds() / 60
                    if not (5 < minutes_until_end < 60):
                        continue

                    # Extract asset
                    asset = "BTC"  # default
                    if "bitcoin" in question.lower():
                        asset = "BTC"
                    elif "ethereum" in question.lower():
                        asset = "ETH"
                    elif "solana" in question.lower():
                        asset = "SOL"
                    elif "xrp" in question.lower():
                        asset = "XRP"

                    updown_markets.append(
                        {
                            "id": market.get("condition_id"),
                            "question": question,
                            "asset": asset,
                            "start_time": start_date_str,
                            "end_time": end_date_str,
                            "duration": minutes_until_end,
                            "volume": market.get("volume", 0),
                            "tokens": market.get("tokens", []),
                        }
                    )

                except Exception as e:
                    continue

            return updown_markets

        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []

    def get_market_outcome(self, market_id: str, end_time_str: str) -> Optional[Dict]:
        """
        Poll for market resolution/outcome.

        Returns dict with actual_end_price and outcome direction if resolved.
        """
        # Check if market should be expired
        try:
            end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)

            # Give 5 minute grace period for resolution
            if now < end_time + timedelta(minutes=5):
                return None  # Not ready yet

            # Fetch market data from API
            url = f"https://gamma-api.polymarket.com/markets/{market_id}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                return None

            market = response.json()

            # Check if closed/resolved
            if not market.get("closed", False):
                return None

            # Extract outcome from tokens/prices
            tokens = market.get("tokens", [])

            # Look for winning token (price close to 1.0)
            for token in tokens:
                outcome_name = token.get("outcome", "").upper()
                price = float(token.get("price", 0))

                if price > 0.95:  # Winner has price ~$1
                    if (
                        "UP" in outcome_name
                        or "YES" in outcome_name
                        or "HIGHER" in outcome_name
                    ):
                        return {"outcome_dir": "UP", "actual_end_price": price}
                    elif (
                        "DOWN" in outcome_name
                        or "NO" in outcome_name
                        or "LOWER" in outcome_name
                    ):
                        return {"outcome_dir": "DOWN", "actual_end_price": price}

            # Fallback: check market outcome description
            # In production, you'd parse resolution data more carefully

            return None  # Couldn't determine outcome yet

        except Exception as e:
            print(f"Error resolving market {market_id}: {e}")
            return None

    def predict_direction(self, asset: str) -> Tuple[Optional[str], float]:
        """
        Get prediction for asset direction.
        Integrates with your predictor module.
        """
        try:
            # Import predictor
            from predict_updown import EnsemblePredictor

            predictor = EnsemblePredictor()
            direction, confidence = predictor.predict(asset)

            return direction, confidence

        except Exception as e:
            print(f"Prediction error for {asset}: {e}")
            # Fallback: no bet
            return None, 0.0

    def calculate_bet_size(self) -> float:
        """
        Calculate bet size using Kelly criterion.

        Uses current win rate and profit factor to determine optimal sizing.
        Defaults to fixed 2% risk if insufficient data.
        """
        if self.total_bets < 20:
            # Not enough data - use fixed risk
            return self.bankroll * self.risk_per_trade

        # Kelly criterion: f = (p*b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio

        win_rate = self.winning_bets / self.total_bets if self.total_bets > 0 else 0.5

        # Get average win/loss from database
        with self.lock:
            avg_win = (
                self.conn.execute(
                    """
                SELECT AVG(virtual_profit) 
                FROM trades 
                WHERE resolved=TRUE AND virtual_profit > 0
            """
                ).fetchone()[0]
                or 10.0
            )

            avg_loss = abs(
                self.conn.execute(
                    """
                SELECT AVG(virtual_profit) 
                FROM trades 
                WHERE resolved=TRUE AND virtual_profit < 0
            """
                ).fetchone()[0]
                or 10.0
            )

        b = avg_win / avg_loss if avg_loss > 0 else 1.0
        kelly = (win_rate * b - (1 - win_rate)) / b

        # Use fractional Kelly (1/4) for safety
        safe_kelly = max(0, kelly * 0.25)

        # Cap at 5% max
        bet_size = min(self.bankroll * safe_kelly, self.bankroll * 0.05)

        return max(bet_size, 5.0)  # Minimum $5 bet

    def place_virtual_bet(self, market: Dict, direction: str, confidence: float):
        """Place a virtual bet on a market."""

        bet_size = self.calculate_bet_size()

        # Don't bet if insufficient bankroll
        if bet_size > self.bankroll:
            print(f"⚠️  Insufficient bankroll: ${self.bankroll:.2f}")
            return

        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO trades 
                (market_id, asset, duration_min, start_time, end_time, 
                 predicted_dir, confidence, virtual_bet_usd, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    market["id"],
                    market["asset"],
                    int(market["duration"]),
                    market["start_time"],
                    market["end_time"],
                    direction,
                    confidence,
                    bet_size,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self.conn.commit()

        print(f"\n📝 [{datetime.now().strftime('%H:%M:%S')}] VIRTUAL BET PLACED")
        print(f"   Market: {market['question']}")
        print(f"   Asset: {market['asset']}")
        print(f"   Direction: {direction}")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Bet size: ${bet_size:.2f}")
        print(f"   Bankroll after: ${self.bankroll:.2f}")

    def resolve_trade(self, trade_row, outcome: Dict):
        """
        Resolve a virtual bet and update P&L.

        Args:
            trade_row: Database row with trade info
            outcome: Dict with actual_end_price and outcome_dir
        """
        market_id = trade_row[1]
        predicted_dir = trade_row[7]
        bet_size = trade_row[9]
        actual_dir = outcome["outcome_dir"]

        # Calculate profit/loss
        # Approximate 80-90% return on winners (after platform fees)
        # Assume fair price at entry (~0.50) for simplicity
        if predicted_dir == actual_dir:
            # Win: paid ~$0.50 per share, get $1.00 back
            # Net profit = bet_size * 0.80 (80% return)
            profit = bet_size * 0.80
        else:
            # Loss: lose entire bet
            profit = -bet_size

        # Update database
        with self.lock:
            self.conn.execute(
                """
                UPDATE trades 
                SET outcome_dir=?, virtual_profit=?, resolved=TRUE, resolved_at=?
                WHERE market_id=?
            """,
                (actual_dir, profit, datetime.now(timezone.utc).isoformat(), market_id),
            )

            # Update bankroll and stats
            self.bankroll += profit
            self.total_bets += 1
            self.total_profit += profit
            if profit > 0:
                self.winning_bets += 1

            # Track max bankroll for drawdown calculation
            if self.bankroll > self.max_bankroll:
                self.max_bankroll = self.bankroll

            self.conn.execute(
                """
                UPDATE portfolio 
                SET virtual_bankroll=?, total_bets=?, winning_bets=?, 
                    total_profit=?, max_bankroll=?, last_updated=?
                WHERE key='main'
            """,
                (
                    self.bankroll,
                    self.total_bets,
                    self.winning_bets,
                    self.total_profit,
                    self.max_bankroll,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            self.conn.commit()

        # Log result
        win_rate = self.winning_bets / self.total_bets if self.total_bets > 0 else 0

        print(f"\n✅ [{datetime.now().strftime('%H:%M:%S')}] TRADE RESOLVED")
        print(f"   Market: {market_id}")
        print(f"   Predicted: {predicted_dir} | Actual: {actual_dir}")
        print(f"   Result: {'WIN' if profit > 0 else 'LOSS'}")
        print(f"   P&L: ${profit:+.2f}")
        print(f"   Bankroll: ${self.bankroll:.2f}")
        print(f"   Total bets: {self.total_bets} | Win rate: {win_rate:.1%}")

    def run_monitor_loop(self):
        """Main event loop - monitors markets and places virtual bets."""

        print("=" * 80)
        print("🤖 PRODUCTION SIMULATOR - PAPER TRADING MODE")
        print("=" * 80)
        print(f"\n💰 Starting bankroll: ${self.bankroll:.2f}")
        print(f"⚙️  Min confidence: {self.min_confidence:.0%}")
        print(f"⚙️  Risk per trade: {self.risk_per_trade:.1%}")
        print(f"⏰ Poll interval: {self.poll_interval}s")
        print("\n🎯 Target: 200+ trades to prove edge")
        print("💡 Press Ctrl+C to stop and see summary\n")

        check_count = 0

        try:
            while True:
                check_count += 1
                timestamp = datetime.now().strftime("%H:%M:%S")

                # 1. Check for new markets
                markets = self.get_open_markets()

                for market in markets:
                    market_id = market["id"]

                    if market_id in self.known_markets:
                        continue

                    self.known_markets.add(market_id)

                    # Get prediction
                    direction, confidence = self.predict_direction(market["asset"])

                    # Place bet if confidence exceeds threshold
                    if direction and confidence >= self.min_confidence:
                        self.place_virtual_bet(market, direction, confidence)
                    else:
                        # Log observation even if no bet
                        with self.lock:
                            self.conn.execute(
                                """
                                INSERT OR IGNORE INTO trades 
                                (market_id, asset, duration_min, start_time, end_time, created_at)
                                VALUES (?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    market_id,
                                    market["asset"],
                                    int(market["duration"]),
                                    market["start_time"],
                                    market["end_time"],
                                    datetime.now(timezone.utc).isoformat(),
                                ),
                            )
                            self.conn.commit()

                # 2. Check for resolutions
                with self.lock:
                    unresolved = self.conn.execute(
                        """
                        SELECT * FROM trades 
                        WHERE resolved=FALSE AND predicted_dir IS NOT NULL
                    """
                    ).fetchall()

                for trade_row in unresolved:
                    market_id = trade_row[1]
                    end_time = trade_row[5]

                    outcome = self.get_market_outcome(market_id, end_time)

                    if outcome:
                        self.resolve_trade(trade_row, outcome)

                # 3. Status update
                if check_count % 10 == 0:
                    unresolved_bets = len(unresolved)
                    print(
                        f"[{timestamp}] Monitoring... ({unresolved_bets} pending, {self.total_bets} resolved)"
                    )

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self.print_summary()

    def print_summary(self):
        """Print comprehensive performance summary."""

        print("\n\n" + "=" * 80)
        print("📊 SIMULATION SUMMARY")
        print("=" * 80)

        win_rate = self.winning_bets / self.total_bets if self.total_bets > 0 else 0
        drawdown = (
            (self.max_bankroll - self.bankroll) / self.max_bankroll
            if self.max_bankroll > 0
            else 0
        )

        print("\n💰 Portfolio:")
        print("   Starting bankroll: $1,000.00")
        print(f"   Current bankroll: ${self.bankroll:.2f}")
        print(
            f"   Total P&L: ${self.total_profit:+.2f} ({self.total_profit/1000*100:+.1f}%)"
        )
        print(f"   Peak bankroll: ${self.max_bankroll:.2f}")
        print(f"   Current drawdown: {drawdown:.1%}")

        print("\n📈 Performance:")
        print(f"   Total bets: {self.total_bets}")
        print(f"   Wins: {self.winning_bets}")
        print(f"   Losses: {self.total_bets - self.winning_bets}")
        print(f"   Win rate: {win_rate:.1%}")

        # Calculate additional metrics
        if self.total_bets > 0:
            with self.lock:
                avg_profit = (
                    self.conn.execute(
                        """
                    SELECT AVG(virtual_profit) FROM trades WHERE resolved=TRUE
                """
                    ).fetchone()[0]
                    or 0
                )

                avg_win = (
                    self.conn.execute(
                        """
                    SELECT AVG(virtual_profit) 
                    FROM trades 
                    WHERE resolved=TRUE AND virtual_profit > 0
                """
                    ).fetchone()[0]
                    or 0
                )

                avg_loss = (
                    self.conn.execute(
                        """
                    SELECT AVG(virtual_profit) 
                    FROM trades 
                    WHERE resolved=TRUE AND virtual_profit < 0
                """
                    ).fetchone()[0]
                    or 0
                )

            print("\n⚖️  Risk Metrics:")
            print(f"   Avg P&L per bet: ${avg_profit:+.2f}")
            print(f"   Avg win: ${avg_win:+.2f}")
            print(f"   Avg loss: ${avg_loss:+.2f}")
            if avg_loss != 0:
                profit_factor = abs(avg_win / avg_loss)
                print(f"   Profit factor: {profit_factor:.2f}")

        # Assessment
        print("\n🎯 Assessment:")
        if self.total_bets < 200:
            print(f"   ⏳ Need {200 - self.total_bets} more bets to assess edge")
        elif win_rate > 0.55 and self.total_profit > 0:
            print("   ✅ Strategy shows promise - strong edge detected")
            print("   💡 Consider tiny live test ($10-20) to validate execution")
        elif win_rate > 0.52:
            print("   ⚠️  Marginal performance - needs improvement")
        else:
            print("   ❌ Strategy underperforming - major changes needed")

        print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys

    min_confidence = float(sys.argv[1]) if len(sys.argv) > 1 else 0.60
    risk_per_trade = float(sys.argv[2]) if len(sys.argv) > 2 else 0.02
    poll_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    sim = Simulator(
        min_confidence=min_confidence,
        risk_per_trade=risk_per_trade,
        poll_interval=poll_interval,
    )

    print("🚀 Starting production simulator")
    print(
        "   Customize: python scripts/monitor_simulator.py [min_conf] [risk] [interval]"
    )
    print("   Example: python scripts/monitor_simulator.py 0.65 0.03 45\n")

    sim.run_monitor_loop()

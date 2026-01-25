#!/usr/bin/env python3
"""
Continuously monitor for new Up or Down markets and log virtual trades.

Features:
- Monitors Gamma API for new Up/Down markets
- Uses SimplePredictor (momentum) to make virtual bets
- Logs all trades to SQLite (virtual_trades.db)
- Tracks outcomes and PnL
"""

import requests
import time
import sqlite3
import os
import sys
import json
from datetime import datetime, timezone
from typing import Optional

# Ensure we can import predict_updown from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from predict_updown import EnsemblePredictor
except ImportError:
    # Fallback if running from root without scripts in path
    try:
        from scripts.predict_updown import EnsemblePredictor
    except ImportError:
        print("Warning: Could not import EnsemblePredictor. Using random stub.")
        EnsemblePredictor = None

DB_PATH = "virtual_trades.db"


def init_db():
    """Initialize SQLite database for virtual trading."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS markets
                 (id TEXT PRIMARY KEY, 
                  asset TEXT, 
                  duration TEXT, 
                  start_price REAL, 
                  predicted_dir TEXT, 
                  confidence REAL,
                  actual_outcome TEXT, 
                  virtual_profit REAL, 
                  status TEXT,
                  timestamp DATETIME,
                  question TEXT)"""
    )
    conn.commit()
    return conn


def get_db_connection():
    return sqlite3.connect(DB_PATH)


def get_market_status(market_id: str) -> dict:
    """Fetch single market details to check outcome."""
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error fetching market {market_id}: {e}")
    return {}


def resolve_market(market_data: dict) -> Optional[str]:
    """Determine the winner from market data."""
    # This logic depends on Gamma API response format for resolved markets
    # Usually 'winner' field or 1.0 price in outcomePrices

    if not market_data.get("closed", False) and market_data.get("active", True):
        return None

    # Check outcome prices
    try:
        outcome_prices = json.loads(market_data.get("outcomePrices", "[]"))
        outcomes = json.loads(market_data.get("outcomes", "[]"))

        if not outcome_prices or not outcomes:
            return None

        # Find which outcome has price 1.0 (or close to 1)
        for i, price in enumerate(outcome_prices):
            if float(price) > 0.95:
                return outcomes[i]
    except:
        pass

    return None


def update_outcomes(conn):
    """Check open bets and resolve them if market finished."""
    c = conn.cursor()
    c.execute("SELECT id, predicted_dir FROM markets WHERE status = 'OPEN'")
    open_markets = c.fetchall()

    if not open_markets:
        return

    print(f"🔎 Checking status of {len(open_markets)} open virtual bets...")

    for market_id, predicted_dir in open_markets:
        market_data = get_market_status(market_id)
        if not market_data:
            continue

        winner = resolve_market(market_data)

        if winner:
            # Determine Profit
            # Assuming Fixed Bet $10. Return $20 if win (roughly, ignoring fees/spread for virtual)
            # Virtual Profit = $10 if Win, -$10 if Loss

            # Normalize winner string (UP/DOWN usually matches outcomes)
            is_win = False
            if winner.upper() == predicted_dir.upper():
                is_win = True

            profit = 10.0 if is_win else -10.0

            c.execute(
                """UPDATE markets 
                         SET actual_outcome = ?, virtual_profit = ?, status = 'CLOSED' 
                         WHERE id = ?""",
                (winner, profit, market_id),
            )
            conn.commit()

            print(
                f"✅ Market {market_id} resolved. Winner: {winner}. Prediction: {predicted_dir}. PnL: ${profit}"
            )


def get_updown_markets():
    """Fetch current Up or Down markets."""
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

            if "up or down" in question.lower():
                end_date_str = market.get("end_date_iso") or market.get("endDate")

                if end_date_str:
                    try:
                        end_date = datetime.fromisoformat(
                            end_date_str.replace("Z", "+00:00")
                        )
                        minutes_until_end = (end_date - now).total_seconds() / 60

                        if 0 < minutes_until_end < 60:
                            updown_markets.append(
                                {
                                    "id": market.get("condition_id")
                                    or market.get("id"),  # Use condition_id or id
                                    "question": question,
                                    "minutes_until_end": minutes_until_end,
                                    "end_date": end_date_str,
                                    "tokens": market.get("tokens", []),
                                    "volume": market.get("volume", 0),
                                }
                            )
                    except:
                        pass

        return updown_markets

    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []


def monitor_markets(interval_seconds=30):
    """Monitor for new Up or Down markets."""

    init_db()
    conn = get_db_connection()

    # Initialize Predictor
    predictor = None
    if EnsemblePredictor:
        try:
            predictor = EnsemblePredictor()
            print("🧠 AI Predictor Initialized (Momentum + Volume)")
        except Exception as e:
            print(f"⚠️ Predictor init failed: {e}")

    print("=" * 80)
    print("🔍 UP/DOWN MARKET MONITOR & VIRTUAL TRADER")
    print("=" * 80)
    print(f"\n⏰ Checking every {interval_seconds} seconds...")
    print("💡 Press Ctrl+C to stop\n")

    check_count = 0

    try:
        while True:
            check_count += 1
            current_time = datetime.now().strftime("%H:%M:%S")

            # 1. Update existing bets
            update_outcomes(conn)

            # 2. Check for new markets
            markets = get_updown_markets()

            if markets:
                # Check for new markets not in DB
                c = conn.cursor()
                c.execute("SELECT id FROM markets")
                existing_ids = {row[0] for row in c.fetchall()}

                new_markets = [m for m in markets if str(m["id"]) not in existing_ids]

                if new_markets:
                    print(f"\n🚨 [{current_time}] NEW MARKETS FOUND!\n")

                    for market in new_markets:
                        print(f"✅ {market['question']}")
                        print(f"   ID: {market['id']}")

                        # Determine Asset
                        asset = "UNKNOWN"
                        if "BTC" in market["question"]:
                            asset = "BTC"
                        elif "ETH" in market["question"]:
                            asset = "ETH"
                        elif "SOL" in market["question"]:
                            asset = "SOL"

                        # Predict
                        direction = "UP"  # Default/Random fallback
                        confidence = 0.0
                        if predictor:
                            pred_dir, conf = predictor.predict(asset)
                            if pred_dir:
                                direction = pred_dir
                                confidence = conf
                                print(
                                    f"   🤖 Model Prediction: {direction} (Conf: {confidence:.1%})"
                                )
                            else:
                                print(f"   🤖 Model Uncertain, defaulting to UP")

                        # Log Virtual Bet
                        try:
                            c.execute(
                                """INSERT INTO markets 
                                         (id, asset, duration, start_price, predicted_dir, confidence, status, timestamp, question, virtual_profit) 
                                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                (
                                    str(market["id"]),
                                    asset,
                                    "Unknown",
                                    0.0,
                                    direction,
                                    confidence,
                                    "OPEN",
                                    datetime.now(),
                                    market["question"],
                                    0.0,
                                ),
                            )
                            conn.commit()
                            print(f"   📝 Virtual Bet LOGGED: {direction} ($10)")
                        except Exception as e:
                            print(f"   ❌ DB Error: {e}")

                        print()

                elif check_count % 10 == 0:  # Status update every 10 checks
                    print(
                        f"[{current_time}] Watching... ({len(markets)} active markets)"
                    )

            else:
                if check_count % 10 == 0:
                    print(f"[{current_time}] No Up/Down markets currently available")

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        conn.close()
        print("\n\n✅ Monitoring stopped.")


if __name__ == "__main__":
    interval = 30
    if len(sys.argv) > 1:
        try:
            interval = int(sys.argv[1])
        except:
            pass

    monitor_markets(interval)

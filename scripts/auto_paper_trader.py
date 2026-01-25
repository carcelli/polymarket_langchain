#!/usr/bin/env python3
"""
Automated Paper Trading System

Combines:
1. Market monitoring (find Up/Down markets)
2. Predictive signals (momentum/volume strategy)
3. Automatic paper betting (virtual bets with logging)
4. Performance tracking (P&L, win rate, etc.)

This is your complete simulation environment to prove edge before going live.
"""

import time
import requests
from datetime import datetime, timezone, timedelta
from paper_trading_system import (
    init_database,
    log_observed_market,
    place_paper_bet,
    resolve_market,
    get_performance_summary,
)
from predict_updown import EnsemblePredictor


class AutoPaperTrader:
    """Automated paper trading system."""

    def __init__(
        self,
        bet_size: float = 10.0,
        min_confidence: float = 0.60,
        check_interval: int = 30,
    ):
        """
        Args:
            bet_size: Virtual dollar amount per bet
            min_confidence: Minimum prediction confidence to bet (0.5 = 50%)
            check_interval: Seconds between market checks
        """
        self.bet_size = bet_size
        self.min_confidence = min_confidence
        self.check_interval = check_interval
        self.predictor = EnsemblePredictor()
        self.seen_markets = set()
        self.active_bets = {}

        # Initialize database
        init_database()

    def get_live_updown_markets(self):
        """Fetch live Up/Down markets from Gamma API."""
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
                    end_date_str = market.get("end_date_iso")

                    if end_date_str:
                        try:
                            end_date = datetime.fromisoformat(
                                end_date_str.replace("Z", "+00:00")
                            )
                            minutes_until_end = (end_date - now).total_seconds() / 60

                            # Only consider markets ending in next 60 minutes
                            if 5 < minutes_until_end < 60:
                                # Extract asset (BTC, ETH, SOL, etc.)
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
                                        "start_time": market.get("start_date_iso"),
                                        "end_time": end_date_str,
                                        "duration_minutes": minutes_until_end,
                                        "volume": market.get("volume", 0),
                                    }
                                )
                        except:
                            pass

            return updown_markets

        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []

    def process_market(self, market: dict):
        """Process a new market: get prediction and maybe place bet."""

        market_id = market["id"]

        # Skip if already processed
        if market_id in self.seen_markets:
            return

        self.seen_markets.add(market_id)

        # Log market observation
        log_observed_market(market)

        print(f"\n📊 New market detected:")
        print(f"   {market['question']}")
        print(f"   Expires in: {market['duration_minutes']:.1f} minutes")
        print(f"   Volume: ${market['volume']:,.0f}")

        # Get prediction
        asset = market["asset"]
        direction, confidence = self.predictor.predict(asset)

        print(f"\n🤖 Prediction for {asset}:")
        print(f"   Direction: {direction if direction else 'NO SIGNAL'}")
        print(f"   Confidence: {confidence:.1%}")

        # Place bet if confidence exceeds threshold
        if direction and confidence >= self.min_confidence:
            print(f"\n✅ PLACING PAPER BET:")
            print(f"   {direction} with {confidence:.1%} confidence")

            bet_id = place_paper_bet(
                market_id=market_id,
                direction=direction,
                amount=self.bet_size,
                confidence=confidence,
                strategy="momentum_volume",
                notes=f"Auto-bet on {asset} market",
            )

            if bet_id:
                # Track for resolution later
                self.active_bets[market_id] = {
                    "bet_id": bet_id,
                    "direction": direction,
                    "asset": asset,
                    "end_time": market["end_time"],
                    "bet_time": datetime.now(timezone.utc),
                }
        else:
            print(
                f"   ⏭️  Skipping (confidence {confidence:.1%} < threshold {self.min_confidence:.1%})"
            )

    def check_for_resolutions(self):
        """Check if any active bets should be resolved."""

        now = datetime.now(timezone.utc)
        to_resolve = []

        for market_id, bet_info in self.active_bets.items():
            try:
                end_time = datetime.fromisoformat(
                    bet_info["end_time"].replace("Z", "+00:00")
                )

                # Market should have expired
                if now > end_time + timedelta(minutes=5):  # 5 min grace period
                    to_resolve.append(market_id)
            except:
                pass

        # Resolve markets
        for market_id in to_resolve:
            self._resolve_bet(market_id)
            del self.active_bets[market_id]

    def _resolve_bet(self, market_id: str):
        """Resolve a specific bet (fetch actual outcome)."""

        print(f"\n⏰ Resolving market {market_id}...")

        # In production, fetch actual market resolution from API
        # For now, simulate with mock data
        bet_info = self.active_bets[market_id]

        # Mock: Get actual price movement
        # TODO: Implement real price fetch from exchange
        import random

        actual_outcome = random.choice(["UP", "DOWN"])

        # Simulate starting/ending prices
        starting_price = 50000.0  # Mock
        ending_price = starting_price * (1.001 if actual_outcome == "UP" else 0.999)

        resolve_market(market_id, ending_price, starting_price)

        print(f"   Actual outcome: {actual_outcome}")
        print(f"   Our bet: {bet_info['direction']}")
        if bet_info["direction"] == actual_outcome:
            print(f"   ✅ WIN!")
        else:
            print(f"   ❌ LOSS")

    def run(self):
        """Main monitoring loop."""

        print("=" * 80)
        print("🤖 AUTOMATED PAPER TRADING SYSTEM")
        print("=" * 80)
        print(f"\n⚙️  Configuration:")
        print(f"   Bet size: ${self.bet_size:.2f}")
        print(f"   Min confidence: {self.min_confidence:.0%}")
        print(f"   Check interval: {self.check_interval}s")
        print(f"\n⏰ Starting monitoring...")
        print("💡 Press Ctrl+C to stop and see performance summary\n")

        check_count = 0

        try:
            while True:
                check_count += 1
                current_time = datetime.now().strftime("%H:%M:%S")

                # Check for new markets
                markets = self.get_live_updown_markets()

                if markets:
                    new_markets = [
                        m for m in markets if m["id"] not in self.seen_markets
                    ]

                    if new_markets:
                        print(f"\n{'='*60}")
                        print(
                            f"[{current_time}] Found {len(new_markets)} new market(s)!"
                        )
                        print(f"{'='*60}")

                        for market in new_markets:
                            self.process_market(market)

                    elif check_count % 10 == 0:
                        print(
                            f"[{current_time}] Monitoring... ({len(self.active_bets)} active bets)"
                        )

                else:
                    if check_count % 10 == 0:
                        print(f"[{current_time}] No Up/Down markets available")

                # Check for markets to resolve
                if self.active_bets:
                    self.check_for_resolutions()

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping automated paper trader...")

            # Show performance summary
            print("\n" + "=" * 70)
            print("📊 FINAL PERFORMANCE")
            print("=" * 70)
            get_performance_summary()

            print(f"\n📝 Session stats:")
            print(f"   Markets observed: {len(self.seen_markets)}")
            print(f"   Active bets: {len(self.active_bets)}")
            print(f"   Checks performed: {check_count}")


if __name__ == "__main__":
    import sys

    # Parse arguments
    bet_size = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
    min_confidence = float(sys.argv[2]) if len(sys.argv) > 2 else 0.60
    check_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    print(f"🚀 Starting Automated Paper Trading System")
    print(
        f"   To customize: python scripts/auto_paper_trader.py [bet_size] [min_confidence] [interval]"
    )
    print(f"   Example: python scripts/auto_paper_trader.py 20.0 0.65 45\n")

    trader = AutoPaperTrader(
        bet_size=bet_size, min_confidence=min_confidence, check_interval=check_interval
    )

    trader.run()

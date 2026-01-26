#!/usr/bin/env python3
"""
NBA Game Simulator - Production Paper Trading for Sports Markets

Combines:
1. Live NBA market discovery (Gamma API)
2. Log5 + home advantage predictor
3. Virtual betting when edge > threshold
4. Outcome tracking and P&L calculation

Sports >>> Crypto for proving edge:
- Longer horizons (games in hours, not 5-15 min)
- Abundant features (records, injuries, rest, venue)
- Documented inefficiencies vs sharp books
"""

import sqlite3
import time
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, List
from nba_market_fetcher import fetch_active_sports_markets, extract_game_info
from nba_predictor import NBAPredictor

DB_PATH = "data/nba_simulator.db"


def init_db():
    """Initialize NBA simulator database."""
    conn = sqlite3.connect(DB_PATH)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nba_bets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT UNIQUE,
            question TEXT,
            matchup TEXT,
            favorite TEXT,
            underdog TEXT,
            home_team TEXT,
            game_date TEXT,
            
            market_price REAL,
            model_prob REAL,
            edge REAL,
            
            bet_team TEXT,
            bet_side TEXT,
            bet_amount REAL,
            bet_time TEXT,
            
            actual_winner TEXT,
            virtual_profit REAL,
            resolved BOOLEAN DEFAULT FALSE,
            resolved_at TEXT
        )
    """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nba_portfolio (
            key TEXT PRIMARY KEY,
            virtual_bankroll REAL,
            total_bets INTEGER DEFAULT 0,
            winning_bets INTEGER DEFAULT 0,
            total_profit REAL DEFAULT 0.0,
            last_updated TEXT
        )
    """
    )

    conn.execute(
        """
        INSERT OR IGNORE INTO nba_portfolio 
        (key, virtual_bankroll, last_updated) 
        VALUES ('main', 1000.0, ?)
    """,
        (datetime.now(timezone.utc).isoformat(),),
    )

    conn.commit()
    return conn


class NBASimulator:
    """Production NBA game outcome simulator."""

    def __init__(
        self,
        min_edge: float = 0.05,
        min_volume: float = 50000,
        bet_amount: float = 20.0,
        poll_interval: int = 300,
    ):
        """
        Args:
            min_edge: Minimum edge to place bet (5% = 0.05)
            min_volume: Minimum market volume ($50k)
            bet_amount: Fixed bet size per game
            poll_interval: Seconds between checks (300 = 5 min)
        """
        self.conn = init_db()
        self.predictor = NBAPredictor()
        self.min_edge = min_edge
        self.min_volume = min_volume
        self.bet_amount = bet_amount
        self.poll_interval = poll_interval

        # Load state
        row = self.conn.execute(
            "SELECT virtual_bankroll, total_bets, winning_bets, total_profit "
            "FROM nba_portfolio WHERE key='main'"
        ).fetchone()

        self.bankroll = row[0]
        self.total_bets = row[1]
        self.winning_bets = row[2]
        self.total_profit = row[3]

        self.known_markets = set()

    def process_market(self, market: Dict):
        """Process new NBA market - analyze and maybe bet."""

        market_id = str(market["id"])

        if market_id in self.known_markets:
            return

        self.known_markets.add(market_id)

        # Skip low volume markets
        if market["volume"] < self.min_volume:
            return

        # Parse game info
        info = extract_game_info(market)

        if not info["favorite"] or not info["underdog"]:
            print(f"⏭️  Skipping: Can't parse teams from '{market['question']}'")
            return

        # Assume first team mentioned is home (enhance with venue parsing later)
        home_team = (
            info["teams_found"][0] if len(info["teams_found"]) > 0 else info["favorite"]
        )
        away_team = (
            info["teams_found"][1] if len(info["teams_found"]) > 1 else info["underdog"]
        )

        # Get prediction
        winner, model_prob, details = self.predictor.predict_winner(
            home_team, away_team, is_team_a_home=True
        )

        # Calculate edge vs market
        market_price = market["yes_price"] if market["yes_price"] else 0.5
        edge, recommendation = self.predictor.calculate_edge(
            home_team, away_team, True, market_price
        )

        print("\n📊 New Market Analyzed:")
        print(f"   {market['question']}")
        print(f"   Matchup: {home_team} vs {away_team}")
        print(f"   Volume: ${market['volume']/1000:.1f}k")
        print(f"   Market: {home_team} at {market_price:.1%}")
        print(f"   Model: {winner} at {model_prob:.1%}")
        print(f"   Edge: {edge:+.1%}")

        # Place bet if edge exceeds threshold
        if abs(edge) >= self.min_edge and recommendation == "BUY":
            self.place_virtual_bet(
                market, home_team, away_team, winner, model_prob, edge, market_price
            )
        else:
            print(f"   ⏭️  PASS: Edge {edge:+.1%} < threshold {self.min_edge:.1%}")

            # Still log observation
            self.conn.execute(
                """
                INSERT OR IGNORE INTO nba_bets 
                (market_id, question, matchup, favorite, underdog, home_team,
                 market_price, model_prob, edge, bet_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    market_id,
                    market["question"],
                    info["matchup"],
                    info["favorite"],
                    info["underdog"],
                    home_team,
                    market_price,
                    model_prob,
                    edge,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            self.conn.commit()

    def place_virtual_bet(
        self,
        market,
        home_team,
        away_team,
        predicted_winner,
        model_prob,
        edge,
        market_price,
    ):
        """Place virtual bet on game."""

        print("\n💰 PLACING VIRTUAL BET:")
        print(f"   Betting on: {predicted_winner}")
        print(f"   Amount: ${self.bet_amount:.2f}")
        print(f"   Edge: {edge:+.1%}")

        info = extract_game_info(market)

        self.conn.execute(
            """
            INSERT OR REPLACE INTO nba_bets 
            (market_id, question, matchup, favorite, underdog, home_team, game_date,
             market_price, model_prob, edge, bet_team, bet_side, bet_amount, bet_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(market["id"]),
                market["question"],
                info["matchup"],
                info["favorite"],
                info["underdog"],
                home_team,
                info["date"],
                market_price,
                model_prob,
                edge,
                predicted_winner,
                "YES" if predicted_winner == home_team else "NO",
                self.bet_amount,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

        print("   ✅ Logged to database")

    def check_resolutions(self):
        """Check for resolved games and update P&L."""

        unresolved = self.conn.execute(
            """
            SELECT market_id, bet_team, bet_amount, question
            FROM nba_bets 
            WHERE resolved=FALSE AND bet_team IS NOT NULL
        """
        ).fetchall()

        if not unresolved:
            return

        print(f"\n🔍 Checking {len(unresolved)} unresolved bets...")

        for market_id, bet_team, bet_amount, question in unresolved:
            # Fetch market status from API
            url = f"https://gamma-api.polymarket.com/markets/{market_id}"

            try:
                response = requests.get(url, timeout=10)
                if response.status_code != 200:
                    continue

                market_data = response.json()

                # Check if closed
                if not market_data.get("closed", False):
                    continue

                # Determine winner from outcome prices
                outcome_prices = market_data.get("outcome_prices", [])
                outcomes = market_data.get("outcomes", [])

                if not outcome_prices or not outcomes:
                    continue

                # Parse prices
                try:
                    if isinstance(outcome_prices, str):
                        prices = [
                            float(p.strip("'\""))
                            for p in outcome_prices.strip("[]").split(",")
                        ]
                    else:
                        prices = [float(p) for p in outcome_prices]

                    if isinstance(outcomes, str):
                        import json

                        outcomes = json.loads(outcomes.replace("'", '"'))

                    # Winner has price ~1.0
                    winner_outcome = None
                    for i, price in enumerate(prices):
                        if price > 0.95:
                            winner_outcome = outcomes[i] if i < len(outcomes) else None
                            break

                    if not winner_outcome:
                        continue

                    # Extract team name from outcome (usually "Team Name" or "Team Name to win")
                    # Simplified: check if bet_team is in winner_outcome
                    is_win = bet_team in winner_outcome

                    # Calculate P&L
                    # Simple model: bet at market price, pay $1 if win
                    # Profit = $1 - entry_price if win, -entry_price if loss
                    # For fixed $20 bet, assume entry at ~50¢: $10 profit if win, -$20 loss
                    profit = bet_amount * 0.80 if is_win else -bet_amount

                    # Update database
                    self.conn.execute(
                        """
                        UPDATE nba_bets
                        SET actual_winner=?, virtual_profit=?, resolved=TRUE, resolved_at=?
                        WHERE market_id=?
                    """,
                        (
                            winner_outcome,
                            profit,
                            datetime.now(timezone.utc).isoformat(),
                            market_id,
                        ),
                    )

                    # Update portfolio
                    self.bankroll += profit
                    self.total_bets += 1
                    self.total_profit += profit
                    if profit > 0:
                        self.winning_bets += 1

                    self.conn.execute(
                        """
                        UPDATE nba_portfolio
                        SET virtual_bankroll=?, total_bets=?, winning_bets=?, 
                            total_profit=?, last_updated=?
                        WHERE key='main'
                    """,
                        (
                            self.bankroll,
                            self.total_bets,
                            self.winning_bets,
                            self.total_profit,
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )

                    self.conn.commit()

                    result = "WIN" if is_win else "LOSS"
                    print(f"✅ {question}")
                    print(f"   Predicted: {bet_team} | Actual: {winner_outcome}")
                    print(f"   Result: {result} | P&L: ${profit:+.2f}")
                    print(f"   Bankroll: ${self.bankroll:.2f}")

                except Exception as e:
                    print(f"Error parsing outcome for {market_id}: {e}")
                    continue

            except Exception as e:
                continue

    def print_summary(self):
        """Print performance summary."""

        print("\n\n" + "=" * 80)
        print("🏀 NBA SIMULATOR SUMMARY")
        print("=" * 80)

        win_rate = self.winning_bets / self.total_bets if self.total_bets > 0 else 0
        roi = self.total_profit / 1000 if self.total_bets > 0 else 0

        print("\n💰 Portfolio:")
        print("   Starting bankroll: $1,000.00")
        print(f"   Current bankroll: ${self.bankroll:.2f}")
        print(f"   Total P&L: ${self.total_profit:+.2f} ({roi:+.1%} ROI)")

        print("\n📈 Performance:")
        print(f"   Total bets: {self.total_bets}")
        print(f"   Wins: {self.winning_bets}")
        print(f"   Losses: {self.total_bets - self.winning_bets}")
        print(f"   Win rate: {win_rate:.1%}")

        if self.total_bets > 0:
            avg_profit = (
                self.conn.execute(
                    """
                SELECT AVG(virtual_profit) FROM nba_bets WHERE resolved=TRUE
            """
                ).fetchone()[0]
                or 0
            )

            print("\n⚖️  Metrics:")
            print(f"   Avg P&L per bet: ${avg_profit:+.2f}")

        print("\n🎯 Assessment:")
        if self.total_bets < 20:
            print(f"   ⏳ Need {20 - self.total_bets} more bets for assessment")
        elif win_rate > 0.55 and self.total_profit > 0:
            print("   ✅ Strong edge detected - outperforming market")
        elif win_rate > 0.52:
            print("   ⚠️  Marginal performance - needs refinement")
        else:
            print("   ❌ Underperforming - revise predictor")

        print("\n" + "=" * 80)

    def run(self):
        """Main monitoring loop."""

        print("=" * 80)
        print("🏀 NBA SIMULATOR - PRODUCTION PAPER TRADING")
        print("=" * 80)
        print(f"\n💰 Starting bankroll: ${self.bankroll:.2f}")
        print(f"⚙️  Min edge: {self.min_edge:.1%}")
        print(f"⚙️  Min volume: ${self.min_volume/1000:.0f}k")
        print(f"⚙️  Bet size: ${self.bet_amount:.2f}")
        print(f"⏰ Poll interval: {self.poll_interval}s")
        print("\n💡 Press Ctrl+C to stop and see summary\n")

        check_count = 0

        try:
            while True:
                check_count += 1
                timestamp = datetime.now().strftime("%H:%M:%S")

                # Check for resolutions
                self.check_resolutions()

                # Fetch new markets
                markets = fetch_active_sports_markets()

                if markets:
                    new_markets = [
                        m for m in markets if str(m["id"]) not in self.known_markets
                    ]

                    if new_markets:
                        print(f"\n{'='*60}")
                        print(
                            f"[{timestamp}] Found {len(new_markets)} new NBA market(s)"
                        )
                        print(f"{'='*60}")

                        for market in new_markets:
                            self.process_market(market)

                    elif check_count % 10 == 0:
                        print(
                            f"[{timestamp}] Monitoring... ({self.total_bets} total bets, {win_rate:.1%} win rate)"
                            if self.total_bets > 0
                            else f"[{timestamp}] Monitoring..."
                        )

                        # Show current win rate
                        if self.total_bets > 0:
                            win_rate = self.winning_bets / self.total_bets

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            self.print_summary()
            self.conn.close()


if __name__ == "__main__":
    import sys

    min_edge = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    min_volume = float(sys.argv[2]) if len(sys.argv) > 2 else 50000
    bet_amount = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    poll_interval = int(sys.argv[4]) if len(sys.argv) > 4 else 300

    sim = NBASimulator(
        min_edge=min_edge,
        min_volume=min_volume,
        bet_amount=bet_amount,
        poll_interval=poll_interval,
    )

    print("🚀 Starting NBA simulator")
    print(
        "   Customize: python scripts/nba_simulator.py [min_edge] [min_vol] [bet_size] [interval]"
    )
    print("   Example: python scripts/nba_simulator.py 0.03 30000 25.0 600\n")

    sim.run()

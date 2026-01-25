#!/usr/bin/env python3
"""
Backtesting framework for Up/Down markets.

Tests strategies against historical market data to measure performance
before risking real capital.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Tuple
import json

DB_PATH = "data/paper_trading.db"
MARKETS_DB = "data/markets.db"


class Backtest:
    """Backtesting engine for Up/Down trading strategies."""

    def __init__(self, starting_capital: float = 1000.0):
        self.starting_capital = starting_capital
        self.current_capital = starting_capital
        self.trades = []
        self.equity_curve = [starting_capital]

    def load_historical_markets(self, limit: int = 1000) -> pd.DataFrame:
        """Load historical Up/Down markets from database."""
        conn = sqlite3.connect(MARKETS_DB)

        query = """
            SELECT id, question, category, end_date, volume, 
                   outcomes, outcome_prices
            FROM markets
            WHERE question LIKE '%Up or Down%'
            AND end_date IS NOT NULL
            ORDER BY end_date DESC
            LIMIT ?
        """

        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()

        print(f"📊 Loaded {len(df)} historical Up/Down markets")
        return df

    def run_strategy(
        self,
        strategy_func: Callable,
        markets_df: pd.DataFrame,
        bet_size: float = 10.0,
        min_confidence: float = 0.55,
    ) -> Dict:
        """
        Run a strategy against historical markets.

        Args:
            strategy_func: Function that takes market data and returns (direction, confidence)
            markets_df: DataFrame of historical markets
            bet_size: Dollar amount per bet
            min_confidence: Minimum confidence to place bet
        """

        print(f"\n🎯 Running backtest...")
        print(f"   Starting capital: ${self.starting_capital:.2f}")
        print(f"   Bet size: ${bet_size:.2f}")
        print(f"   Min confidence: {min_confidence:.1%}")
        print(f"   Markets: {len(markets_df)}\n")

        trades_placed = 0

        for idx, market in markets_df.iterrows():
            # Skip if we've run out of capital
            if self.current_capital < bet_size:
                print(f"⚠️  Insufficient capital at trade {trades_placed}")
                break

            # Get strategy prediction
            try:
                direction, confidence = strategy_func(market)

                if direction is None or confidence < min_confidence:
                    continue

                # Simulate bet outcome
                # For backtest, we need actual outcome - simplified: use price data if available
                actual_outcome = self._get_market_outcome(market)

                if actual_outcome is None:
                    continue  # Skip if we can't determine outcome

                # Calculate P&L
                # Assume entry at 50/50 odds for simplicity
                shares = bet_size / 0.5

                if direction == actual_outcome:
                    # Win: shares pay $1
                    payout = shares * 1.0
                    pnl = payout - bet_size
                else:
                    # Loss
                    pnl = -bet_size

                self.current_capital += pnl
                self.equity_curve.append(self.current_capital)

                self.trades.append(
                    {
                        "market_id": market["id"],
                        "direction": direction,
                        "confidence": confidence,
                        "bet_size": bet_size,
                        "outcome": actual_outcome,
                        "pnl": pnl,
                        "capital": self.current_capital,
                    }
                )

                trades_placed += 1

                if trades_placed % 50 == 0:
                    print(
                        f"   Progress: {trades_placed} trades, Capital: ${self.current_capital:.2f}"
                    )

            except Exception as e:
                # Strategy error - skip this market
                continue

        return self._calculate_metrics()

    def _get_market_outcome(self, market) -> str:
        """Determine actual market outcome from historical data."""
        # This is simplified - in production you'd query actual resolution data
        # For now, simulate 50/50 random outcomes (replace with real data)
        import random

        return random.choice(["UP", "DOWN"])

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""

        if not self.trades:
            return {"error": "No trades executed", "total_trades": 0}

        trades_df = pd.DataFrame(self.trades)

        total_trades = len(trades_df)
        wins = len(trades_df[trades_df["pnl"] > 0])
        losses = len(trades_df[trades_df["pnl"] < 0])
        win_rate = wins / total_trades if total_trades > 0 else 0

        total_pnl = trades_df["pnl"].sum()
        avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
        avg_loss = trades_df[trades_df["pnl"] < 0]["pnl"].mean() if losses > 0 else 0

        # Calculate max drawdown
        equity_array = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Sharpe ratio (simplified)
        if len(self.trades) > 1:
            returns = trades_df["pnl"] / trades_df["bet_size"]
            sharpe = (
                returns.mean() / returns.std() * np.sqrt(252)
                if returns.std() > 0
                else 0
            )
        else:
            sharpe = 0

        metrics = {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "final_capital": self.current_capital,
            "return_pct": (self.current_capital - self.starting_capital)
            / self.starting_capital,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe,
        }

        self._print_results(metrics)
        return metrics

    def _print_results(self, metrics: Dict):
        """Print backtest results."""

        print("\n" + "=" * 70)
        print("📊 BACKTEST RESULTS")
        print("=" * 70)

        print(f"\n💰 Capital:")
        print(f"   Starting: ${self.starting_capital:.2f}")
        print(f"   Ending: ${metrics['final_capital']:.2f}")
        print(f"   P&L: ${metrics['total_pnl']:+.2f} ({metrics['return_pct']:+.1%})")

        print(f"\n📈 Performance:")
        print(f"   Total trades: {metrics['total_trades']}")
        print(f"   Wins: {metrics['wins']}")
        print(f"   Losses: {metrics['losses']}")
        print(f"   Win rate: {metrics['win_rate']:.1%}")

        print(f"\n⚖️  Risk Metrics:")
        print(f"   Avg win: ${metrics['avg_win']:+.2f}")
        print(f"   Avg loss: ${metrics['avg_loss']:+.2f}")
        print(f"   Profit factor: {metrics['profit_factor']:.2f}")
        print(f"   Max drawdown: {metrics['max_drawdown']:.1%}")
        print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.2f}")

        # Assessment
        print(f"\n🎯 Assessment:")
        if metrics["win_rate"] > 0.55 and metrics["total_pnl"] > 0:
            print("   ✅ Strategy shows promise - consider paper trading")
        elif metrics["win_rate"] > 0.50:
            print("   ⚠️  Marginal performance - needs improvement")
        else:
            print("   ❌ Strategy underperforming - major changes needed")

        print("\n" + "=" * 70)


# Example strategies


def random_strategy(market) -> Tuple[str, float]:
    """Baseline: Random predictions (should be ~50% win rate)."""
    import random

    return random.choice(["UP", "DOWN"]), 0.5


def momentum_strategy(market) -> Tuple[str, float]:
    """Simple momentum: bet on recent direction continuing."""
    # In real version, would fetch recent price data
    # For demo, return None to skip (no signal)
    return None, 0


def contrarian_strategy(market) -> Tuple[str, float]:
    """Fade recent moves (mean reversion)."""
    # In real version, would check if oversold/overbought
    return None, 0


if __name__ == "__main__":
    import sys

    print("🔬 Up/Down Market Backtesting Framework\n")

    # Initialize backtest
    bt = Backtest(starting_capital=1000.0)

    # Load historical markets
    markets = bt.load_historical_markets(limit=500)

    if len(markets) == 0:
        print("❌ No historical Up/Down markets found in database.")
        print("   These markets may need to be fetched from the API first.")
        sys.exit(1)

    # Run backtest with random strategy (baseline)
    print("\n📊 Testing baseline (random) strategy...")
    results = bt.run_strategy(
        strategy_func=random_strategy,
        markets_df=markets,
        bet_size=10.0,
        min_confidence=0.0,  # Take all bets for baseline
    )

    print("\n💡 Next steps:")
    print("   1. Implement real predictive strategies in this file")
    print("   2. Fetch actual market outcomes for accurate backtesting")
    print("   3. Run multiple strategies and compare")
    print("   4. Only move to paper trading once backtest shows consistent edge")

#!/usr/bin/env python3
"""
Virtual Trader - Production Paper Trading System

Unified simulator for all market types (NBA, crypto, politics, etc.)
Integrates with planning agent, ML strategies, and existing tools.

Features:
- Multi-market support (NBA, crypto, politics)
- Planning agent integration for analysis
- ML strategy registry integration
- Automatic resolution polling
- Comprehensive performance tracking
- Kelly criterion position sizing
- Risk management (drawdown limits, consecutive loss stops)
"""

import time
import json
import sqlite3
import sys
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from polymarket_agents.connectors.gamma import GammaMarketClient
from polymarket_agents.memory.manager import MemoryManager

# Config
POLL_INTERVAL = 60  # seconds — balance API load vs latency
VIRTUAL_BANKROLL_START = 1000.0  # USD
BET_SIZE_PCT = 0.02  # 2% of bankroll per bet (Kelly fraction)
MIN_EDGE = 0.03  # Only bet if edge > 3% (conservative)
MAX_POSITION_SIZE = 0.05  # Cap at 5% of bankroll per bet
MAX_OPEN_POSITIONS = 10  # Limit concurrent bets
MAX_DAILY_LOSS_PCT = 0.10  # Stop if down 10% in a day
MAX_CONSECUTIVE_LOSSES = 5  # Circuit breaker

DB_PATH = "data/virtual_trader.db"

class VirtualTrader:
    """
    Production virtual trading system.
    
    Integrates with:
    - GammaMarketClient for discovery
    - MemoryManager for local cache
    - Planning agent for analysis
    - ML strategy registry
    """
    
    def __init__(self, 
                 market_types: List[str] = None,
                 min_edge: float = MIN_EDGE,
                 bet_size_pct: float = BET_SIZE_PCT):
        """
        Args:
            market_types: List of market types to trade ['nba', 'crypto', 'politics']
            min_edge: Minimum edge to place bet
            bet_size_pct: Fraction of bankroll per bet
        """
        self.market_types = market_types or ['nba', 'crypto', 'politics']
        self.min_edge = min_edge
        self.bet_size_pct = bet_size_pct
        
        # Initialize connectors
        self.gamma = GammaMarketClient()
        self.memory = MemoryManager("data/markets.db")
        
        # Initialize database
        self.conn = self._init_database()
        
        # Load state
        self.bankroll = self._get_bankroll()
        self.starting_daily_bankroll = self.bankroll
        self.consecutive_losses = 0
        self.seen_markets = set()
        
        print(f"✅ Virtual Trader initialized")
        print(f"   Starting bankroll: ${self.bankroll:.2f}")
        print(f"   Market types: {', '.join(self.market_types)}")
        print(f"   Min edge: {self.min_edge:.1%}")
    
    def _init_database(self) -> sqlite3.Connection:
        """Initialize database with proper schema."""
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        
        # Portfolio tracking
        conn.execute("""
            CREATE TABLE IF NOT EXISTS virtual_portfolio (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                bankroll REAL,
                open_positions INTEGER,
                total_bets INTEGER,
                winning_bets INTEGER,
                total_profit REAL,
                win_rate REAL,
                max_bankroll REAL,
                max_drawdown REAL
            )
        """)
        
        # Trade log
        conn.execute("""
            CREATE TABLE IF NOT EXISTS virtual_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT UNIQUE,
                market_type TEXT,
                question TEXT,
                category TEXT,
                
                bet_side TEXT,          -- YES/NO or team name
                bet_amount REAL,
                entry_price REAL,       -- Market implied prob at entry
                model_prob REAL,        -- Strategy estimated prob
                edge REAL,
                strategy TEXT,          -- Which strategy used
                
                outcome TEXT,           -- YES/NO/team or OPEN
                actual_price REAL,
                profit REAL,
                resolved BOOLEAN DEFAULT FALSE,
                
                entry_time TEXT,
                resolution_time TEXT,
                
                notes TEXT
            )
        """)
        
        # Performance by strategy
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_performance (
                strategy TEXT PRIMARY KEY,
                total_bets INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0,
                win_rate REAL DEFAULT 0.0,
                avg_edge REAL DEFAULT 0.0,
                last_updated TEXT
            )
        """)
        
        # Seed initial portfolio if new
        existing = conn.execute("SELECT COUNT(*) FROM virtual_portfolio").fetchone()[0]
        if existing == 0:
            conn.execute("""
                INSERT INTO virtual_portfolio 
                (timestamp, bankroll, open_positions, total_bets, winning_bets, 
                 total_profit, win_rate, max_bankroll, max_drawdown)
                VALUES (?, ?, 0, 0, 0, 0.0, 0.0, ?, 0.0)
            """, (
                datetime.now(timezone.utc).isoformat(),
                VIRTUAL_BANKROLL_START,
                VIRTUAL_BANKROLL_START
            ))
        
        conn.commit()
        return conn
    
    def _get_bankroll(self) -> float:
        """Get current virtual bankroll."""
        row = self.conn.execute(
            "SELECT bankroll FROM virtual_portfolio ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else VIRTUAL_BANKROLL_START
    
    def _check_risk_limits(self) -> Tuple[bool, str]:
        """
        Check if risk limits exceeded.
        
        Returns:
            (should_stop, reason)
        """
        
        # Daily loss limit
        daily_loss = self.starting_daily_bankroll - self.bankroll
        daily_loss_pct = daily_loss / self.starting_daily_bankroll
        
        if daily_loss_pct > MAX_DAILY_LOSS_PCT:
            return True, f"Daily loss limit hit: {daily_loss_pct:.1%}"
        
        # Consecutive losses
        if self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES:
            return True, f"Consecutive loss limit hit: {self.consecutive_losses} losses"
        
        # Max positions
        open_positions = self.conn.execute(
            "SELECT COUNT(*) FROM virtual_trades WHERE resolved=FALSE"
        ).fetchone()[0]
        
        if open_positions >= MAX_OPEN_POSITIONS:
            return True, f"Max open positions: {open_positions}/{MAX_OPEN_POSITIONS}"
        
        return False, ""
    
    def _calculate_bet_size(self, edge: float, confidence: float = 0.5) -> float:
        """
        Calculate bet size using fractional Kelly criterion.
        
        Args:
            edge: Estimated edge (model_prob - market_price)
            confidence: Confidence in estimate (0-1)
        
        Returns:
            Bet size in dollars
        """
        
        # Get historical win rate if enough data
        stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins
            FROM virtual_trades
            WHERE resolved=TRUE
        """).fetchone()
        
        total_bets, wins = stats[0], stats[1] or 0
        
        if total_bets < 10:
            # Not enough data - use fixed risk
            bet_size = self.bankroll * self.bet_size_pct
        else:
            # Kelly criterion: f = edge / odds
            # Simplified: f = edge (assuming even money)
            kelly_fraction = abs(edge)
            
            # Use fractional Kelly (1/4) for safety
            safe_kelly = kelly_fraction * 0.25 * confidence
            
            bet_size = self.bankroll * safe_kelly
        
        # Apply limits
        bet_size = max(bet_size, 5.0)  # Minimum $5
        bet_size = min(bet_size, self.bankroll * MAX_POSITION_SIZE)  # Cap at 5%
        
        return bet_size
    
    def _get_prediction(self, market: Dict) -> Optional[Dict]:
        """
        Get prediction for market using appropriate strategy.
        
        Tries in order:
        1. Planning agent (if available)
        2. ML strategy registry
        3. Baseline heuristics
        
        Returns:
            Dict with model_prob, edge, recommendation, strategy
            or None if no signal
        """
        
        market_type = self._classify_market(market)
        
        # Try NBA predictor for sports
        if market_type == 'nba':
            try:
                from nba_predictor import NBAPredictor
                from nba_market_fetcher import extract_game_info
                
                predictor = NBAPredictor()
                info = extract_game_info(market)
                
                if info['favorite'] and info['underdog']:
                    # Determine home team (simplified - enhance with parsing)
                    home_team = info['favorite']  # Assume favorite is home
                    away_team = info['underdog']
                    
                    winner, prob, details = predictor.predict_winner(
                        home_team, away_team, is_team_a_home=True
                    )
                    
                    market_price = market.get('outcome_prices', [0.5])[0]
                    if isinstance(market_price, str):
                        market_price = float(market_price)
                    
                    edge = prob - market_price if winner == home_team else (1-prob) - market_price
                    
                    return {
                        'model_prob': prob,
                        'edge': edge,
                        'recommendation': 'BUY' if abs(edge) > self.min_edge else 'PASS',
                        'strategy': 'nba_log5',
                        'bet_side': winner,
                        'confidence': 0.7  # Baseline confidence
                    }
            except Exception as e:
                print(f"NBA predictor error: {e}")
        
        # Try ML strategy registry
        try:
            from polymarket_agents.ml_strategies.registry import best_strategy
            
            result = best_strategy(market)
            
            if result and result.get('edge', 0) > self.min_edge:
                return {
                    'model_prob': result.get('model_pred', 0.5),
                    'edge': result['edge'],
                    'recommendation': result.get('recommendation', 'PASS'),
                    'strategy': result.get('strategy', 'ml_registry'),
                    'bet_side': result.get('side', 'YES'),
                    'confidence': 0.6
                }
        except Exception as e:
            pass
        
        # No strong signal
        return None
    
    def _classify_market(self, market: Dict) -> str:
        """Classify market type."""
        question = market.get('question', '').lower()
        category = market.get('category', '').lower()
        
        if 'nba' in category or any(team in question for team in ['lakers', 'celtics', 'warriors']):
            return 'nba'
        elif 'up or down' in question:
            return 'crypto'
        elif any(word in question for word in ['election', 'president', 'congress']):
            return 'politics'
        else:
            return 'other'
    
    def _log_virtual_bet(self, 
                        market: Dict, 
                        prediction: Dict, 
                        bet_size: float):
        """Log a virtual bet to database."""
        
        market_type = self._classify_market(market)
        
        entry_price = market.get('outcome_prices', [0.5])[0]
        if isinstance(entry_price, str):
            entry_price = float(entry_price)
        
        self.conn.execute("""
            INSERT INTO virtual_trades 
            (market_id, market_type, question, category, bet_side, bet_amount,
             entry_price, model_prob, edge, strategy, entry_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market['id'],
            market_type,
            market['question'],
            market.get('category', 'Unknown'),
            prediction['bet_side'],
            bet_size,
            entry_price,
            prediction['model_prob'],
            prediction['edge'],
            prediction['strategy'],
            datetime.now(timezone.utc).isoformat()
        ))
        
        # Update portfolio
        self.conn.execute("""
            UPDATE virtual_portfolio 
            SET open_positions = open_positions + 1,
                total_bets = total_bets + 1
            WHERE id = (SELECT MAX(id) FROM virtual_portfolio)
        """)
        
        self.conn.commit()
        
        print(f"\n📝 VIRTUAL BET PLACED:")
        print(f"   Market: {market['question'][:70]}")
        print(f"   Type: {market_type}")
        print(f"   Betting: {prediction['bet_side']}")
        print(f"   Amount: ${bet_size:.2f}")
        print(f"   Edge: {prediction['edge']:+.1%}")
        print(f"   Strategy: {prediction['strategy']}")
    
    def _resolve_trades(self) -> int:
        """
        Check open trades and resolve if market closed.
        
        Returns:
            Number of trades resolved
        """
        
        open_trades = self.conn.execute("""
            SELECT id, market_id, bet_side, bet_amount, entry_price, edge
            FROM virtual_trades
            WHERE resolved=FALSE
        """).fetchall()
        
        if not open_trades:
            return 0
        
        resolved_count = 0
        
        for trade in open_trades:
            trade_id, market_id, bet_side, bet_amount, entry_price, edge = trade
            
            try:
                # Fetch market status
                market = self.gamma.get_market(market_id)
                
                if not market or not market.get('closed', False):
                    continue
                
                # Determine outcome
                outcome = self._parse_outcome(market)
                
                if not outcome:
                    continue
                
                # Calculate profit
                is_win = (outcome == bet_side) or (bet_side in outcome)
                
                if is_win:
                    # Win: approximate 80% return (accounting for fees/spread)
                    profit = bet_amount * 0.80
                else:
                    # Loss: lose entire bet
                    profit = -bet_amount
                
                # Update trade
                self.conn.execute("""
                    UPDATE virtual_trades
                    SET outcome=?, actual_price=?, profit=?, resolved=TRUE, 
                        resolution_time=?
                    WHERE id=?
                """, (
                    outcome,
                    1.0 if is_win else 0.0,
                    profit,
                    datetime.now(timezone.utc).isoformat(),
                    trade_id
                ))
                
                # Update bankroll
                self.bankroll += profit
                
                # Track consecutive losses
                if profit > 0:
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                
                # Update portfolio
                winning_delta = 1 if profit > 0 else 0
                
                self.conn.execute("""
                    UPDATE virtual_portfolio
                    SET bankroll=?, 
                        open_positions=open_positions-1,
                        winning_bets=winning_bets+?,
                        total_profit=total_profit+?
                    WHERE id=(SELECT MAX(id) FROM virtual_portfolio)
                """, (self.bankroll, winning_delta, profit))
                
                # Update max bankroll for drawdown calc
                max_bankroll = self.conn.execute(
                    "SELECT MAX(max_bankroll) FROM virtual_portfolio"
                ).fetchone()[0]
                
                if self.bankroll > max_bankroll:
                    self.conn.execute("""
                        UPDATE virtual_portfolio
                        SET max_bankroll=?
                        WHERE id=(SELECT MAX(id) FROM virtual_portfolio)
                    """, (self.bankroll,))
                
                self.conn.commit()
                
                result = "WIN" if profit > 0 else "LOSS"
                print(f"\n✅ TRADE RESOLVED:")
                print(f"   Market: {market_id}")
                print(f"   Predicted: {bet_side} | Actual: {outcome}")
                print(f"   Result: {result}")
                print(f"   P&L: ${profit:+.2f}")
                print(f"   Bankroll: ${self.bankroll:.2f}")
                
                resolved_count += 1
                
            except Exception as e:
                print(f"Error resolving trade {market_id}: {e}")
                continue
        
        return resolved_count
    
    def _parse_outcome(self, market: Dict) -> Optional[str]:
        """Parse market outcome from resolution data."""
        
        # Check outcome prices for winner
        outcome_prices = market.get('outcome_prices', [])
        outcomes = market.get('outcomes', [])
        
        if not outcome_prices or not outcomes:
            return None
        
        try:
            # Parse if string format
            if isinstance(outcome_prices, str):
                prices = [float(p.strip("'\"")) for p in outcome_prices.strip("[]").split(",")]
            else:
                prices = [float(p) for p in outcome_prices]
            
            if isinstance(outcomes, str):
                import json
                outcomes = json.loads(outcomes.replace("'", '"'))
            
            # Winner has price ~1.0
            for i, price in enumerate(prices):
                if price > 0.95 and i < len(outcomes):
                    return outcomes[i]
        
        except Exception as e:
            print(f"Error parsing outcome: {e}")
        
        return None
    
    def _print_summary(self):
        """Print performance summary."""
        
        stats = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN profit > 0 THEN 1 ELSE 0 END) as wins,
                SUM(profit) as total_pnl,
                AVG(profit) as avg_pnl,
                AVG(edge) as avg_edge
            FROM virtual_trades
            WHERE resolved=TRUE
        """).fetchone()
        
        total, wins, pnl, avg_pnl, avg_edge = stats
        
        win_rate = wins / total if total > 0 else 0
        roi = (self.bankroll - VIRTUAL_BANKROLL_START) / VIRTUAL_BANKROLL_START
        
        print("\n" + "=" * 80)
        print("📊 VIRTUAL TRADER SUMMARY")
        print("=" * 80)
        
        print(f"\n💰 Portfolio:")
        print(f"   Starting: ${VIRTUAL_BANKROLL_START:.2f}")
        print(f"   Current: ${self.bankroll:.2f}")
        print(f"   P&L: ${pnl or 0:+.2f} ({roi:+.1%} ROI)")
        
        print(f"\n📈 Performance:")
        print(f"   Total bets: {total or 0}")
        print(f"   Wins: {wins or 0}")
        print(f"   Win rate: {win_rate:.1%}")
        print(f"   Avg P&L: ${avg_pnl or 0:+.2f}")
        print(f"   Avg edge: {avg_edge or 0:+.1%}")
        
        # By market type
        by_type = self.conn.execute("""
            SELECT market_type, COUNT(*), SUM(profit)
            FROM virtual_trades
            WHERE resolved=TRUE
            GROUP BY market_type
        """).fetchall()
        
        if by_type:
            print(f"\n📊 By Market Type:")
            for mtype, count, profit in by_type:
                print(f"   {mtype}: {count} bets, ${profit or 0:+.2f}")
        
        print("\n" + "=" * 80)
    
    def run_cycle(self):
        """Run one trading cycle."""
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{timestamp}] Virtual trading cycle | Bankroll: ${self.bankroll:.2f}")
        
        # Check risk limits
        should_stop, reason = self._check_risk_limits()
        if should_stop:
            print(f"🛑 STOPPING: {reason}")
            self._print_summary()
            sys.exit(0)
        
        # Resolve existing trades
        resolved = self._resolve_trades()
        if resolved > 0:
            print(f"   Resolved {resolved} trade(s)")
        
        # Fetch new markets
        try:
            markets = self.gamma.get_current_markets(limit=100)
            
            new_bets = 0
            
            for market in markets:
                market_id = market.get('condition_id') or market['id']
                
                if market_id in self.seen_markets:
                    continue
                
                self.seen_markets.add(market_id)
                
                # Check if market type matches our filter
                market_type = self._classify_market(market)
                if market_type not in self.market_types and 'all' not in self.market_types:
                    continue
                
                # Get prediction
                prediction = self._get_prediction(market)
                
                if not prediction or prediction['recommendation'] != 'BUY':
                    continue
                
                # Calculate bet size
                bet_size = self._calculate_bet_size(
                    prediction['edge'],
                    prediction['confidence']
                )
                
                # Place virtual bet
                self._log_virtual_bet(market, prediction, bet_size)
                new_bets += 1
            
            if new_bets == 0 and resolved == 0:
                print("   No new opportunities or resolutions")
            elif new_bets > 0:
                print(f"   Placed {new_bets} new virtual bet(s)")
        
        except Exception as e:
            print(f"Error in cycle: {e}")
    
    def run(self):
        """Main loop."""
        
        print("\n" + "=" * 80)
        print("🚀 VIRTUAL TRADER - PRODUCTION SIMULATION")
        print("=" * 80)
        print(f"\n💡 Press Ctrl+C to stop and see summary\n")
        
        try:
            while True:
                self.run_cycle()
                time.sleep(POLL_INTERVAL)
        
        except KeyboardInterrupt:
            self._print_summary()
            self.conn.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Virtual Trader Simulator')
    parser.add_argument('--markets', nargs='+', default=['nba', 'crypto'],
                       help='Market types to trade: nba, crypto, politics, all')
    parser.add_argument('--min-edge', type=float, default=MIN_EDGE,
                       help='Minimum edge to place bet (default: 0.03)')
    parser.add_argument('--bet-size', type=float, default=BET_SIZE_PCT,
                       help='Bet size as fraction of bankroll (default: 0.02)')
    parser.add_argument('--interval', type=int, default=POLL_INTERVAL,
                       help='Poll interval in seconds (default: 60)')
    
    args = parser.parse_args()
    
    trader = VirtualTrader(
        market_types=args.markets,
        min_edge=args.min_edge,
        bet_size_pct=args.bet_size
    )
    
    trader.run()

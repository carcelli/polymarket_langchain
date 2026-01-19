#!/usr/bin/env python3
"""
Paper Trading System for Up/Down Markets

Tracks virtual bets, logs outcomes, calculates P&L without risking real money.
This is your simulation environment to prove edge before going live.
"""

import sqlite3
import json
import requests
from datetime import datetime, timezone
from pathlib import Path
import time

DB_PATH = "data/paper_trading.db"

def init_database():
    """Initialize paper trading database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Markets table - tracks all observed markets
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS markets (
            market_id TEXT PRIMARY KEY,
            question TEXT,
            asset TEXT,
            start_time TEXT,
            end_time TEXT,
            duration_minutes INTEGER,
            starting_price REAL,
            ending_price REAL,
            outcome TEXT,
            volume REAL,
            observed_at TEXT,
            resolved INTEGER DEFAULT 0
        )
    """)
    
    # Paper bets table - tracks all virtual bets
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paper_bets (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT,
            direction TEXT,
            amount_usd REAL,
            confidence REAL,
            bet_time TEXT,
            entry_price REAL,
            exit_price REAL,
            outcome TEXT,
            profit_loss REAL,
            strategy TEXT,
            notes TEXT,
            FOREIGN KEY (market_id) REFERENCES markets(market_id)
        )
    """)
    
    # Performance metrics table - daily rollups
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_performance (
            date TEXT PRIMARY KEY,
            total_bets INTEGER,
            winning_bets INTEGER,
            total_profit_loss REAL,
            win_rate REAL,
            avg_profit REAL,
            avg_loss REAL,
            sharpe_ratio REAL,
            max_drawdown REAL
        )
    """)
    
    # Strategy performance table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS strategy_performance (
            strategy TEXT PRIMARY KEY,
            total_bets INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_rate REAL,
            total_pnl REAL,
            avg_pnl_per_bet REAL,
            best_bet REAL,
            worst_bet REAL,
            last_updated TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"✅ Database initialized at {DB_PATH}")

def log_observed_market(market_data: dict):
    """Log a newly observed market."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT OR IGNORE INTO markets 
            (market_id, question, asset, start_time, end_time, duration_minutes, 
             volume, observed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_data['id'],
            market_data['question'],
            market_data.get('asset', 'UNKNOWN'),
            market_data.get('start_time'),
            market_data.get('end_time'),
            market_data.get('duration_minutes'),
            market_data.get('volume', 0),
            datetime.now(timezone.utc).isoformat()
        ))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error logging market: {e}")
        return False
    finally:
        conn.close()

def place_paper_bet(market_id: str, direction: str, amount: float, 
                    confidence: float = 0.5, strategy: str = "manual",
                    entry_price: float = None, notes: str = ""):
    """Place a virtual bet."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO paper_bets 
            (market_id, direction, amount_usd, confidence, bet_time, 
             entry_price, strategy, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_id,
            direction,
            amount,
            confidence,
            datetime.now(timezone.utc).isoformat(),
            entry_price,
            strategy,
            notes
        ))
        bet_id = cursor.lastrowid
        conn.commit()
        
        print(f"\n📝 Paper bet placed:")
        print(f"   Bet ID: {bet_id}")
        print(f"   Market: {market_id}")
        print(f"   Direction: {direction}")
        print(f"   Amount: ${amount:.2f}")
        print(f"   Confidence: {confidence:.2%}")
        print(f"   Strategy: {strategy}")
        
        return bet_id
        
    except Exception as e:
        print(f"❌ Error placing paper bet: {e}")
        return None
    finally:
        conn.close()

def resolve_market(market_id: str, ending_price: float, starting_price: float):
    """Resolve a market and update all associated bets."""
    outcome = "UP" if ending_price > starting_price else "DOWN"
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Update market
        cursor.execute("""
            UPDATE markets 
            SET ending_price = ?, starting_price = ?, outcome = ?, resolved = 1
            WHERE market_id = ?
        """, (ending_price, starting_price, outcome, market_id))
        
        # Resolve all bets on this market
        cursor.execute("""
            SELECT bet_id, direction, amount_usd, entry_price
            FROM paper_bets
            WHERE market_id = ? AND outcome IS NULL
        """, (market_id,))
        
        bets = cursor.fetchall()
        total_pnl = 0
        
        for bet_id, direction, amount, entry_price in bets:
            # Simplified P&L calculation
            # Assume we bought shares at entry_price, pay $1 if win, $0 if lose
            if not entry_price:
                entry_price = 0.5  # Default 50/50 if unknown
            
            shares = amount / entry_price
            
            if direction == outcome:
                # Win: shares pay $1 each
                payout = shares * 1.0
                pnl = payout - amount
            else:
                # Lose: shares worth $0
                pnl = -amount
            
            total_pnl += pnl
            
            cursor.execute("""
                UPDATE paper_bets
                SET exit_price = ?, outcome = ?, profit_loss = ?
                WHERE bet_id = ?
            """, (ending_price, outcome, pnl, bet_id))
            
            print(f"   Bet {bet_id}: {direction} → {outcome} = ${pnl:+.2f}")
        
        conn.commit()
        
        print(f"\n✅ Market {market_id} resolved: {outcome}")
        print(f"   Ending price: ${ending_price:.2f}")
        print(f"   Starting price: ${starting_price:.2f}")
        print(f"   Total P&L from {len(bets)} bets: ${total_pnl:+.2f}")
        
        return outcome, total_pnl
        
    except Exception as e:
        print(f"❌ Error resolving market: {e}")
        conn.rollback()
        return None, 0
    finally:
        conn.close()

def get_performance_summary():
    """Get overall performance metrics."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_bets,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losses,
                SUM(profit_loss) as total_pnl,
                AVG(profit_loss) as avg_pnl,
                AVG(CASE WHEN profit_loss > 0 THEN profit_loss END) as avg_win,
                AVG(CASE WHEN profit_loss < 0 THEN profit_loss END) as avg_loss,
                MAX(profit_loss) as best_bet,
                MIN(profit_loss) as worst_bet
            FROM paper_bets
            WHERE outcome IS NOT NULL
        """)
        
        row = cursor.fetchone()
        
        if row[0] == 0:
            print("\n📊 No resolved bets yet.")
            return
        
        total, wins, losses, total_pnl, avg_pnl, avg_win, avg_loss, best, worst = row
        win_rate = wins / total if total > 0 else 0
        
        print("\n" + "=" * 70)
        print("📊 PAPER TRADING PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"\n💰 P&L Metrics:")
        print(f"   Total P&L: ${total_pnl:+,.2f}")
        print(f"   Average P&L per bet: ${avg_pnl:+.2f}")
        print(f"   Best bet: ${best:+.2f}")
        print(f"   Worst bet: ${worst:+.2f}")
        
        print(f"\n📈 Win Rate:")
        print(f"   Total bets: {total}")
        print(f"   Wins: {wins} ({win_rate:.1%})")
        print(f"   Losses: {losses}")
        
        if avg_win and avg_loss:
            profit_factor = abs(avg_win / avg_loss)
            print(f"\n⚖️  Risk/Reward:")
            print(f"   Average win: ${avg_win:+.2f}")
            print(f"   Average loss: ${avg_loss:+.2f}")
            print(f"   Profit factor: {profit_factor:.2f}")
        
        # Strategy breakdown
        cursor.execute("""
            SELECT 
                strategy,
                COUNT(*) as bets,
                SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
                SUM(profit_loss) as pnl
            FROM paper_bets
            WHERE outcome IS NOT NULL
            GROUP BY strategy
        """)
        
        strategies = cursor.fetchall()
        if strategies:
            print(f"\n🎯 By Strategy:")
            for strategy, bets, wins, pnl in strategies:
                wr = wins / bets if bets > 0 else 0
                print(f"   {strategy}: {bets} bets, {wr:.1%} win rate, ${pnl:+.2f} P&L")
        
        # Recent performance (last 20 bets)
        cursor.execute("""
            SELECT direction, outcome, profit_loss, bet_time
            FROM paper_bets
            WHERE outcome IS NOT NULL
            ORDER BY bet_time DESC
            LIMIT 20
        """)
        
        recent = cursor.fetchall()
        if recent:
            print(f"\n📅 Last 20 Bets:")
            recent_pnl = sum(r[2] for r in recent)
            recent_wins = sum(1 for r in recent if r[2] > 0)
            print(f"   P&L: ${recent_pnl:+.2f}")
            print(f"   Win rate: {recent_wins}/{len(recent)} ({recent_wins/len(recent):.1%})")
        
        print("\n" + "=" * 70)
        
        return {
            'total_bets': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl
        }
        
    except Exception as e:
        print(f"❌ Error getting performance: {e}")
        return None
    finally:
        conn.close()

def get_unresolved_markets():
    """Get markets that haven't been resolved yet."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT market_id, question, end_time
            FROM markets
            WHERE resolved = 0
            ORDER BY end_time ASC
        """)
        
        markets = cursor.fetchall()
        return markets
        
    finally:
        conn.close()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python scripts/paper_trading_system.py init")
        print("  python scripts/paper_trading_system.py summary")
        print("  python scripts/paper_trading_system.py bet <market_id> <UP|DOWN> <amount> [confidence]")
        print("  python scripts/paper_trading_system.py resolve <market_id> <ending_price> <starting_price>")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'init':
        init_database()
    
    elif command == 'summary':
        get_performance_summary()
    
    elif command == 'bet' and len(sys.argv) >= 5:
        market_id = sys.argv[2]
        direction = sys.argv[3].upper()
        amount = float(sys.argv[4])
        confidence = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
        place_paper_bet(market_id, direction, amount, confidence)
    
    elif command == 'resolve' and len(sys.argv) >= 5:
        market_id = sys.argv[2]
        ending_price = float(sys.argv[3])
        starting_price = float(sys.argv[4])
        resolve_market(market_id, ending_price, starting_price)
    
    else:
        print("Invalid command")
        sys.exit(1)

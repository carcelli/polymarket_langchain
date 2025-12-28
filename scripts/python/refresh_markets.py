#!/usr/bin/env python3
"""
Polymarket Database Refresh Service

Continuously updates the local markets database with fresh data from Polymarket.

Usage:
    # One-time refresh
    python scripts/python/refresh_markets.py
    
    # Continuous refresh (every 5 minutes)
    python scripts/python/refresh_markets.py --continuous --interval 300
    
    # Background daemon
    nohup python scripts/python/refresh_markets.py --continuous --interval 300 &

Terminology:
    - Market: A single prediction question (e.g., "Will Bitcoin reach $100k?")
    - Event: A collection of related markets (e.g., "Super Bowl 2026" with 32 team markets)
"""

import argparse
import sqlite3
import json
import time
import sys
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any

import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

GAMMA_URL = "https://gamma-api.polymarket.com"
DB_PATH = "data/markets.db"

# Category detection
CATEGORIES = {
    "politics": {
        "tags": ["Trump", "Election", "Congress", "Senate", "President", "Democrat", "Republican", "Biden", "Political"],
        "keywords": ["trump", "biden", "election", "senate", "congress", "president", "democrat", "republican", "vote", "governor", "primary"],
    },
    "sports": {
        "tags": ["Sports", "NFL", "NBA", "MLB", "NHL", "Soccer", "UFC", "MMA", "Football", "Basketball", "Tennis"],
        "keywords": ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", "championship", "super bowl", "playoffs", "world series"],
    },
    "crypto": {
        "tags": ["Crypto", "Bitcoin", "Ethereum", "DeFi", "NFT"],
        "keywords": ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", "defi", "solana", "token"],
    },
    "tech": {
        "tags": ["Tech", "AI", "Apple", "Google", "Microsoft", "OpenAI"],
        "keywords": ["ai", "tech", "apple", "google", "openai", "software", "startup", "silicon valley"],
    },
    "geopolitics": {
        "tags": ["War", "Ukraine", "Russia", "China", "International", "Conflict"],
        "keywords": ["ukraine", "russia", "china", "war", "ceasefire", "conflict", "nato", "military"],
    },
    "culture": {
        "tags": ["Culture", "Entertainment", "Music", "Movies", "Celebrity", "Awards"],
        "keywords": ["celebrity", "movie", "music", "entertainment", "award", "oscar", "grammy", "tv show"],
    },
    "finance": {
        "tags": ["Finance", "Stock", "Fed", "Interest", "Market"],
        "keywords": ["stock", "fed", "interest rate", "market", "finance", "wall street", "s&p", "nasdaq"],
    },
    "economy": {
        "tags": ["Economy", "GDP", "Jobs", "Inflation", "Recession"],
        "keywords": ["gdp", "jobs", "inflation", "economic", "recession", "unemployment"],
    },
    "science": {
        "tags": ["Health", "Science", "Space", "Climate", "Medical"],
        "keywords": ["health", "science", "space", "climate", "research", "nasa", "medical", "disease"],
    },
}


def categorize_event(event: Dict) -> str:
    """Determine category based on tags and keywords."""
    event_tags = [t.get('label', t.get('name', '')).lower() for t in event.get('tags', [])]
    title = (event.get('title', '') + ' ' + event.get('description', '')).lower()
    
    scores = defaultdict(int)
    for cat_id, cat_info in CATEGORIES.items():
        for tag in cat_info['tags']:
            if tag.lower() in event_tags:
                scores[cat_id] += 3
        for keyword in cat_info['keywords']:
            if keyword in title:
                scores[cat_id] += 1
    
    if scores:
        return max(scores, key=scores.get)
    return 'other'


def init_db(db_path: str = DB_PATH):
    """Initialize database schema."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
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
    """)
    
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON markets(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_active ON markets(active)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_volume ON markets(volume)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_last_updated ON markets(last_updated)")
    
    conn.commit()
    conn.close()


def store_market(cursor, market: Dict, category: str, event_id: str):
    """Store a single market in the database."""
    outcomes = market.get('outcomes', '[]')
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except:
            outcomes = []
    
    prices = market.get('outcomePrices', '[]')
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except:
            prices = []
    
    cursor.execute("""
        INSERT OR REPLACE INTO markets 
        (id, question, description, category, outcomes, outcome_prices, volume, liquidity,
         active, end_date, slug, clob_token_ids, event_id, tags, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(market.get('id')),
        market.get('question'),
        market.get('description', ''),
        category,
        json.dumps(outcomes) if isinstance(outcomes, list) else outcomes,
        json.dumps(prices) if isinstance(prices, list) else prices,
        float(market.get('volume', 0) or 0),
        float(market.get('liquidity', 0) or 0),
        1 if market.get('active') else 0,
        market.get('endDate'),
        market.get('slug', ''),
        market.get('clobTokenIds', ''),
        str(event_id),
        json.dumps([t.get('label', '') for t in market.get('tags', [])]),
        datetime.now().isoformat()
    ))


def fetch_events(limit: int = 100, offset: int = 0) -> List[Dict]:
    """Fetch events from Polymarket API."""
    try:
        response = httpx.get(
            f"{GAMMA_URL}/events",
            params={
                'closed': 'false',
                'limit': limit,
                'offset': offset,
                'order': 'id',
                'ascending': 'false'
            },
            timeout=60
        )
        return response.json()
    except Exception as e:
        print(f"   ⚠️  Error fetching events: {str(e)[:50]}")
        return []


def cleanup_expired_markets(db_path: str = DB_PATH, grace_hours: int = 24) -> Dict[str, int]:
    """
    Remove expired markets from the database.
    
    Args:
        db_path: Path to SQLite database
        grace_hours: Hours after end_date before deletion (default: 24)
        
    Returns:
        Dict with cleanup statistics
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Count expired markets
    cursor.execute("""
        SELECT COUNT(*), COALESCE(SUM(volume), 0)
        FROM markets 
        WHERE end_date IS NOT NULL 
        AND end_date != ''
        AND datetime(end_date) < datetime('now', ?)
    """, (f'-{grace_hours} hours',))
    
    result = cursor.fetchone()
    expired_count = result[0] or 0
    expired_volume = result[1] or 0
    
    if expired_count > 0:
        # Get category breakdown
        cursor.execute("""
            SELECT category, COUNT(*) 
            FROM markets 
            WHERE end_date IS NOT NULL 
            AND end_date != ''
            AND datetime(end_date) < datetime('now', ?)
            GROUP BY category
        """, (f'-{grace_hours} hours',))
        expired_by_category = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Delete expired markets
        cursor.execute("""
            DELETE FROM markets 
            WHERE end_date IS NOT NULL 
            AND end_date != ''
            AND datetime(end_date) < datetime('now', ?)
        """, (f'-{grace_hours} hours',))
        
        conn.commit()
    else:
        expired_by_category = {}
    
    conn.close()
    
    return {
        "deleted_count": expired_count,
        "deleted_volume": expired_volume,
        "deleted_by_category": expired_by_category
    }


def refresh_database(max_events: int = 500, db_path: str = DB_PATH, cleanup: bool = True, grace_hours: int = 24) -> Dict[str, int]:
    """
    Refresh the database with latest markets.
    
    Args:
        max_events: Maximum events to fetch
        db_path: Database path
        cleanup: Whether to remove expired markets (default: True)
        grace_hours: Hours after end_date before deletion (default: 24)
    
    Returns:
        Dict with refresh statistics
    """
    print(f"\n{'='*60}")
    print(f"  🔄 MARKET DATABASE REFRESH")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    init_db(db_path)
    
    # Step 1: Cleanup expired markets
    if cleanup:
        print(f"\n  🧹 Cleaning up expired markets (>{grace_hours}h past end_date)...")
        cleanup_stats = cleanup_expired_markets(db_path, grace_hours)
        
        if cleanup_stats["deleted_count"] > 0:
            print(f"     Deleted: {cleanup_stats['deleted_count']:,} markets (${cleanup_stats['deleted_volume']:,.0f} volume)")
            for cat, count in cleanup_stats["deleted_by_category"].items():
                print(f"       - {cat}: {count}")
        else:
            print(f"     No expired markets to clean up")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get current stats
    cursor.execute("SELECT COUNT(*) FROM markets WHERE active = 1")
    start_count = cursor.fetchone()[0]
    
    stats = {
        "start_count": start_count,
        "events_fetched": 0,
        "markets_updated": 0,
        "new_markets": 0,
        "categories": defaultdict(int),
    }
    
    print(f"\n  📊 Current: {start_count:,} active markets")
    print(f"  📥 Fetching up to {max_events} events...\n")
    
    # Fetch events in batches
    offset = 0
    batch_size = 50
    
    while offset < max_events:
        events = fetch_events(limit=batch_size, offset=offset)
        
        if not events:
            break
        
        stats["events_fetched"] += len(events)
        
        for event in events:
            category = categorize_event(event)
            event_id = event.get('id', '')
            
            for market in event.get('markets', []):
                if market.get('active'):
                    store_market(cursor, market, category, event_id)
                    stats["markets_updated"] += 1
                    stats["categories"][category] += 1
        
        conn.commit()
        
        # Progress indicator
        print(f"  ✓ Offset {offset}: {len(events)} events, {stats['markets_updated']} markets", end='\r')
        
        offset += batch_size
        time.sleep(0.3)  # Rate limiting
    
    # Get final stats
    cursor.execute("SELECT COUNT(*) FROM markets WHERE active = 1")
    end_count = cursor.fetchone()[0]
    stats["end_count"] = end_count
    stats["new_markets"] = end_count - start_count
    
    cursor.execute("SELECT SUM(volume) FROM markets WHERE active = 1")
    stats["total_volume"] = cursor.fetchone()[0] or 0
    
    conn.close()
    
    # Print summary
    print(f"\n\n  {'─'*56}")
    print(f"  ✅ REFRESH COMPLETE")
    print(f"  {'─'*56}")
    if cleanup:
        print(f"  Expired deleted: {cleanup_stats['deleted_count']:,}")
    print(f"  Events fetched:  {stats['events_fetched']:,}")
    print(f"  Markets updated: {stats['markets_updated']:,}")
    print(f"  Total markets:   {stats['end_count']:,}")
    print(f"  Net change:      {stats['new_markets']:+,}")
    print(f"  Total volume:    ${stats['total_volume']:,.0f}")
    print(f"\n  By Category:")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count:,}")
    print(f"{'='*60}\n")
    
    if cleanup:
        stats["cleanup"] = cleanup_stats
    
    return stats


def run_continuous(interval: int = 300, max_events: int = 500, db_path: str = DB_PATH, 
                   cleanup: bool = True, grace_hours: int = 24):
    """
    Run continuous refresh loop.
    
    Args:
        interval: Seconds between refreshes (default: 300 = 5 minutes)
        max_events: Max events to fetch per refresh
        db_path: Database path
        cleanup: Whether to remove expired markets
        grace_hours: Hours after expiration before deletion
    """
    print(f"\n🔁 Starting continuous refresh (every {interval}s)")
    print(f"   Auto-cleanup: {'ON' if cleanup else 'OFF'} (grace: {grace_hours}h)")
    print(f"   Press Ctrl+C to stop\n")
    
    refresh_count = 0
    
    try:
        while True:
            refresh_count += 1
            print(f"\n[Refresh #{refresh_count}]")
            
            try:
                refresh_database(
                    max_events=max_events, 
                    db_path=db_path,
                    cleanup=cleanup,
                    grace_hours=grace_hours
                )
            except Exception as e:
                print(f"   ❌ Refresh failed: {str(e)}")
            
            print(f"   💤 Next refresh in {interval}s...")
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Stopped after {refresh_count} refreshes")


def main():
    parser = argparse.ArgumentParser(
        description="Refresh Polymarket database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # One-time refresh with cleanup
    python refresh_markets.py
    
    # Refresh without cleanup (keep expired markets)
    python refresh_markets.py --no-cleanup
    
    # Refresh with custom grace period (48 hours)
    python refresh_markets.py --grace-hours 48
    
    # Continuous refresh every 5 minutes
    python refresh_markets.py --continuous --interval 300
    
    # Background daemon
    nohup python refresh_markets.py --continuous &
        """
    )
    
    parser.add_argument('--continuous', '-c', action='store_true',
                        help='Run continuous refresh loop')
    parser.add_argument('--interval', '-i', type=int, default=300,
                        help='Seconds between refreshes (default: 300)')
    parser.add_argument('--max-events', '-m', type=int, default=500,
                        help='Max events to fetch per refresh (default: 500)')
    parser.add_argument('--db-path', '-d', type=str, default=DB_PATH,
                        help='Database path')
    parser.add_argument('--no-cleanup', action='store_true',
                        help='Skip removal of expired markets')
    parser.add_argument('--grace-hours', '-g', type=int, default=24,
                        help='Hours after expiration before deletion (default: 24)')
    
    args = parser.parse_args()
    
    cleanup = not args.no_cleanup
    
    if args.continuous:
        run_continuous(
            interval=args.interval,
            max_events=args.max_events,
            db_path=args.db_path,
            cleanup=cleanup,
            grace_hours=args.grace_hours
        )
    else:
        refresh_database(
            max_events=args.max_events,
            db_path=args.db_path,
            cleanup=cleanup,
            grace_hours=args.grace_hours
        )


if __name__ == "__main__":
    main()


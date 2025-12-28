#!/usr/bin/env python3
"""
Multi-Category Markets Workflow
================================

This script fetches Polymarket markets across all categories and stores them
in a SQLite database with proper categorization.

Categories:
- Politics (US Politics, Elections, Trump, Congress)
- Sports (NFL, NBA, Soccer, etc.)
- Crypto (Bitcoin, Ethereum, DeFi)
- Finance (Stocks, Fed, Interest Rates)
- Tech (AI, Big Tech, Startups)
- Geopolitics (Ukraine, Middle East, China)
- Culture (Celebrities, Music, Movies)
- Science (Health, Space, Climate)
- Economy (GDP, Inflation, Jobs)

Usage:
    python scripts/python/category_workflow.py                    # All categories
    python scripts/python/category_workflow.py --category crypto  # Single category
    python scripts/python/category_workflow.py --list             # List categories
    python scripts/python/category_workflow.py --stats            # Show DB stats
"""

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict

import httpx

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================================
# CATEGORY DEFINITIONS
# ============================================================================

CATEGORIES = {
    "politics": {
        "name": "Politics",
        "emoji": "🏛️",
        "description": "US Politics, Elections, Government",
        "tags": ["Politics", "U.S. Politics", "Trump", "Trump Presidency", "elections 2024", 
                 "USA Election", "US Election", "Senate", "Congress", "democratic presidential nomination",
                 "vice president", "GOP", "DNC"],
        "keywords": ["trump", "biden", "election", "president", "congress", "senate", 
                     "democrat", "republican", "governor", "vote", "ballot", "political",
                     "white house", "cabinet", "impeach", "campaign", "nominee"],
    },
    "sports": {
        "name": "Sports",
        "emoji": "⚽",
        "description": "NFL, NBA, Soccer, MMA, Olympics",
        "tags": ["Sports", "NFL", "NBA", "MLB", "NHL", "Soccer", "UFC", "MMA", "Olympics",
                 "Football", "Basketball", "Baseball", "Tennis", "Golf", "F1"],
        "keywords": ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball", 
                     "baseball", "hockey", "tennis", "golf", "ufc", "mma", "boxing",
                     "championship", "playoffs", "super bowl", "world series", "finals"],
    },
    "crypto": {
        "name": "Crypto",
        "emoji": "₿",
        "description": "Bitcoin, Ethereum, DeFi, NFTs",
        "tags": ["Crypto", "Crypto Prices", "Bitcoin", "Ethereum", "DeFi", "NFT"],
        "keywords": ["bitcoin", "btc", "ethereum", "eth", "crypto", "blockchain", 
                     "defi", "nft", "solana", "altcoin", "token", "wallet", "exchange"],
    },
    "finance": {
        "name": "Finance",
        "emoji": "📈",
        "description": "Stocks, Fed, Interest Rates, Markets",
        "tags": ["Finance", "Stock Market", "Fed", "Interest Rates", "IPO", "Earnings"],
        "keywords": ["stock", "fed", "interest rate", "inflation", "bank", "trading",
                     "investment", "earnings", "ipo", "nasdaq", "s&p", "dow", "market"],
    },
    "tech": {
        "name": "Tech",
        "emoji": "💻",
        "description": "AI, Big Tech, Startups",
        "tags": ["Tech", "AI", "Big Tech", "DeepSeek", "OpenAI"],
        "keywords": ["ai", "artificial intelligence", "tech", "apple", "google", 
                     "microsoft", "meta", "amazon", "tesla", "openai", "chatgpt",
                     "software", "startup", "elon musk", "zuckerberg"],
    },
    "geopolitics": {
        "name": "Geopolitics",
        "emoji": "🌍",
        "description": "Ukraine, Middle East, China, Conflicts",
        "tags": ["Geopolitics", "World", "Ukraine", "russia", "Middle East", "Israel",
                 "Foreign Policy", "China", "NATO"],
        "keywords": ["war", "ukraine", "russia", "china", "nato", "military", "nuclear",
                     "iran", "israel", "korea", "conflict", "invasion", "ceasefire",
                     "sanctions", "diplomacy"],
    },
    "culture": {
        "name": "Culture",
        "emoji": "🎬",
        "description": "Celebrities, Music, Movies, Entertainment",
        "tags": ["Culture", "Celebrities", "Music", "Movies", "Entertainment", "Awards"],
        "keywords": ["celebrity", "movie", "music", "entertainment", "tv", "award",
                     "oscar", "grammy", "kardashian", "twitter", "tiktok", "youtube",
                     "netflix", "spotify", "concert", "album"],
    },
    "science": {
        "name": "Science",
        "emoji": "🔬",
        "description": "Health, Space, Climate, Research",
        "tags": ["Science", "Health", "Space", "Climate", "NASA", "FDA"],
        "keywords": ["science", "space", "nasa", "climate", "health", "covid", 
                     "vaccine", "fda", "research", "medicine", "disease", "pandemic"],
    },
    "economy": {
        "name": "Economy",
        "emoji": "💰",
        "description": "GDP, Jobs, Inflation, Economic Policy",
        "tags": ["Economy", "Economic Policy", "GDP", "Jobs", "Inflation", "Recession"],
        "keywords": ["economy", "gdp", "jobs", "unemployment", "recession", "growth",
                     "trade", "tariff", "deficit", "debt", "fiscal"],
    },
}

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

DB_PATH = "data/markets.db"

def init_database():
    """Initialize the categorized markets database."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Main markets table with category
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
    
    # Category summary table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            name TEXT PRIMARY KEY,
            display_name TEXT,
            emoji TEXT,
            description TEXT,
            market_count INTEGER DEFAULT 0,
            last_updated TEXT
        )
    """)
    
    # Index for faster category queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON markets(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_active ON markets(active)")
    
    conn.commit()
    conn.close()
    
    # Initialize category records
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    for cat_id, cat_info in CATEGORIES.items():
        cursor.execute("""
            INSERT OR REPLACE INTO categories (name, display_name, emoji, description, last_updated)
            VALUES (?, ?, ?, ?, ?)
        """, (cat_id, cat_info["name"], cat_info["emoji"], cat_info["description"], 
              datetime.now().isoformat()))
    conn.commit()
    conn.close()


def store_market(market: Dict, category: str):
    """Store a single market in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Parse JSON fields if needed
    outcomes = market.get('outcomes', [])
    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except:
            pass
    
    prices = market.get('outcomePrices', [])
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except:
            pass
    
    clob_ids = market.get('clobTokenIds', [])
    if isinstance(clob_ids, str):
        try:
            clob_ids = json.loads(clob_ids)
        except:
            pass
    
    tags = [t.get('label', t.get('name', '')) for t in market.get('tags', [])]
    
    cursor.execute("""
        INSERT OR REPLACE INTO markets 
        (id, question, description, category, outcomes, outcome_prices, volume, 
         liquidity, active, end_date, slug, clob_token_ids, event_id, tags, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        str(market.get('id')),
        market.get('question'),
        market.get('description', ''),
        category,
        json.dumps(outcomes),
        json.dumps(prices),
        float(market.get('volume', 0) or 0),
        float(market.get('liquidity', 0) or 0),
        1 if market.get('active') else 0,
        market.get('endDate'),
        market.get('slug', ''),
        json.dumps(clob_ids),
        str(market.get('eventId', '')),
        json.dumps(tags),
        datetime.now().isoformat()
    ))
    
    conn.commit()
    conn.close()


def update_category_counts():
    """Update market counts for each category."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        UPDATE categories SET market_count = (
            SELECT COUNT(*) FROM markets WHERE markets.category = categories.name
        ), last_updated = ?
    """, (datetime.now().isoformat(),))
    
    conn.commit()
    conn.close()


def get_category_stats() -> List[Dict]:
    """Get statistics for all categories."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT c.name, c.display_name, c.emoji, c.description,
               COUNT(m.id) as total_markets,
               SUM(CASE WHEN m.active = 1 THEN 1 ELSE 0 END) as active_markets,
               SUM(m.volume) as total_volume
        FROM categories c
        LEFT JOIN markets m ON c.name = m.category
        GROUP BY c.name
        ORDER BY total_markets DESC
    """)
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(r) for r in rows]


def get_markets_by_category(category: str, limit: int = 20) -> List[Dict]:
    """Get markets for a specific category."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT * FROM markets 
        WHERE category = ? AND active = 1
        ORDER BY volume DESC
        LIMIT ?
    """, (category, limit))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(r) for r in rows]


# ============================================================================
# API FUNCTIONS
# ============================================================================

GAMMA_API_URL = "https://gamma-api.polymarket.com"


def categorize_event(event: Dict) -> str:
    """Determine the category for an event based on its tags and title."""
    event_tags = [t.get('label', t.get('name', '')).lower() for t in event.get('tags', [])]
    title = (event.get('title', '') + ' ' + event.get('description', '')).lower()
    
    # Score each category
    scores = defaultdict(int)
    
    for cat_id, cat_info in CATEGORIES.items():
        # Check tags
        for tag in cat_info['tags']:
            if tag.lower() in event_tags:
                scores[cat_id] += 3
        
        # Check keywords in title/description
        for keyword in cat_info['keywords']:
            if keyword in title:
                scores[cat_id] += 1
    
    # Return highest scoring category, or 'other' if no match
    if scores:
        return max(scores, key=scores.get)
    return 'other'


def fetch_events_by_category(category: str, limit: int = 30) -> List[Dict]:
    """Fetch events that match a specific category."""
    cat_info = CATEGORIES.get(category)
    if not cat_info:
        return []
    
    all_events = []
    
    # Try fetching by tag names
    for tag in cat_info['tags'][:5]:  # Limit to avoid too many API calls
        try:
            # Search for events with this tag
            params = {
                "active": True,
                "closed": False,
                "limit": limit // 3,
            }
            response = httpx.get(f"{GAMMA_API_URL}/events", params=params, timeout=30)
            if response.status_code == 200:
                events = response.json()
                # Filter events that have matching tags
                for event in events:
                    event_tags = [t.get('label', '').lower() for t in event.get('tags', [])]
                    if tag.lower() in event_tags:
                        all_events.append(event)
            time.sleep(0.2)
        except Exception as e:
            print(f"    Error fetching tag '{tag}': {e}")
    
    # Deduplicate
    seen_ids = set()
    unique_events = []
    for event in all_events:
        eid = event.get('id')
        if eid and eid not in seen_ids:
            seen_ids.add(eid)
            unique_events.append(event)
    
    return unique_events[:limit]


def fetch_all_active_events(limit: int = 200) -> List[Dict]:
    """Fetch all active events and categorize them."""
    all_events = []
    offset = 0
    batch_size = 50
    
    while len(all_events) < limit:
        try:
            params = {
                "active": True,
                "closed": False,
                "limit": batch_size,
                "offset": offset,
            }
            response = httpx.get(f"{GAMMA_API_URL}/events", params=params, timeout=30)
            if response.status_code == 200:
                events = response.json()
                if not events:
                    break
                all_events.extend(events)
                offset += batch_size
            else:
                break
            time.sleep(0.3)
        except Exception as e:
            print(f"Error fetching events at offset {offset}: {e}")
            break
    
    return all_events[:limit]


# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_category_stats(stats: List[Dict]):
    """Print category statistics."""
    print_header("MARKET CATEGORIES")
    print()
    
    total_markets = sum(s['total_markets'] or 0 for s in stats)
    total_volume = sum(s['total_volume'] or 0 for s in stats)
    
    for stat in stats:
        emoji = stat['emoji']
        name = stat['display_name']
        count = stat['total_markets'] or 0
        active = stat['active_markets'] or 0
        volume = stat['total_volume'] or 0
        
        bar_len = min(int(count / max(total_markets, 1) * 30), 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        
        print(f"  {emoji} {name:<15} │ {bar} │ {count:>4} markets │ ${volume:>12,.0f}")
    
    print()
    print(f"  {'TOTAL':<18} │ {'':30} │ {total_markets:>4} markets │ ${total_volume:>12,.0f}")
    print()


def print_markets_table(markets: List[Dict], category: str):
    """Print markets in a formatted table."""
    cat_info = CATEGORIES.get(category, {"emoji": "📊", "name": category.title()})
    
    print_header(f"{cat_info['emoji']} {cat_info['name'].upper()} MARKETS")
    print()
    
    if not markets:
        print("  No markets found.")
        return
    
    for i, m in enumerate(markets[:10]):
        question = m.get('question', 'N/A')[:55]
        volume = m.get('volume', 0) or 0
        
        try:
            prices = json.loads(m.get('outcome_prices', '[]'))
            if len(prices) >= 2:
                yes = float(prices[0]) * 100
                odds = f"YES: {yes:5.1f}%"
            else:
                odds = "N/A"
        except:
            odds = "N/A"
        
        print(f"  {i+1:>2}. {question}")
        print(f"      {odds} │ Vol: ${volume:,.0f}")
        print()


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def run_full_ingestion(limit: int = 100, categories_filter: List[str] = None):
    """Run the full categorized ingestion workflow."""
    start_time = datetime.now()
    
    print_header("POLYMARKET CATEGORIZED INGESTION")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Limit: {limit} events")
    
    # Initialize database
    print("\n  📁 Initializing database...")
    init_database()
    
    # Fetch all active events
    print("\n  📥 Fetching active events from Polymarket...")
    events = fetch_all_active_events(limit=limit)
    print(f"     Found {len(events)} events")
    
    # Categorize and store
    print("\n  🏷️  Categorizing markets...")
    category_counts = defaultdict(int)
    
    for event in events:
        category = categorize_event(event)
        
        # Skip if filtering and not in filter
        if categories_filter and category not in categories_filter:
            continue
        
        # Get markets from event
        markets = event.get('markets', [])
        for market in markets:
            if market.get('active') and not market.get('closed'):
                # Add event tags to market
                market['tags'] = event.get('tags', [])
                market['eventId'] = event.get('id')
                
                store_market(market, category)
                category_counts[category] += 1
    
    # Update counts
    update_category_counts()
    
    # Print summary
    print("\n  ✅ Ingestion complete!")
    print("\n  Category breakdown:")
    for cat_id, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        cat_info = CATEGORIES.get(cat_id, {"emoji": "📊", "name": cat_id})
        print(f"     {cat_info['emoji']} {cat_info['name']}: {count} markets")
    
    total = sum(category_counts.values())
    duration = (datetime.now() - start_time).total_seconds()
    
    print(f"\n  📊 Total: {total} markets stored in {duration:.1f}s")
    
    return category_counts


def show_category_list():
    """Show available categories."""
    print_header("AVAILABLE CATEGORIES")
    print()
    for cat_id, cat_info in CATEGORIES.items():
        print(f"  {cat_info['emoji']} {cat_id:<12} - {cat_info['name']}")
        print(f"     {cat_info['description']}")
        print()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Category Polymarket Ingestion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--category", "-c", type=str, help="Ingest only this category")
    parser.add_argument("--limit", "-l", type=int, default=100, help="Max events to fetch")
    parser.add_argument("--list", action="store_true", help="List available categories")
    parser.add_argument("--stats", action="store_true", help="Show database statistics")
    parser.add_argument("--view", type=str, help="View markets in a category")
    
    args = parser.parse_args()
    
    if args.list:
        show_category_list()
    elif args.stats:
        init_database()
        stats = get_category_stats()
        print_category_stats(stats)
    elif args.view:
        init_database()
        markets = get_markets_by_category(args.view)
        print_markets_table(markets, args.view)
    else:
        categories_filter = [args.category] if args.category else None
        run_full_ingestion(limit=args.limit, categories_filter=categories_filter)


if __name__ == "__main__":
    main()


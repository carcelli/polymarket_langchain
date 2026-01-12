#!/usr/bin/env python3
"""
Politics Markets Workflow
=========================

This script demonstrates the complete workflow for:
1. Fetching political markets from Polymarket API
2. Filtering and visualizing the data
3. Storing markets in the SQLite database

Workflow Diagram:
┌─────────────────────────────────────────────────────────────────────────┐
│                     POLYMARKET POLITICS WORKFLOW                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐   │
│   │   Gamma API  │────▶│   Filter     │────▶│  Store in Database   │   │
│   │  (Politics)  │     │  Active Only │     │     (SQLite)         │   │
│   └──────────────┘     └──────────────┘     └──────────────────────┘   │
│          │                    │                       │                 │
│          ▼                    ▼                       ▼                 │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────────────┐   │
│   │  Raw Markets │     │ Tradeable    │     │  markets table       │   │
│   │  JSON Data   │     │ Markets      │     │  news table          │   │
│   └──────────────┘     └──────────────┘     └──────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Usage:
    python scripts/python/politics_workflow.py [--limit 20] [--dry-run]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import httpx

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from polymarket_agents.memory.manager import MemoryManager


# ============================================================================
# CONSTANTS
# ============================================================================

GAMMA_API_URL = "https://gamma-api.polymarket.com"

# Politics-related tag IDs (verified from Polymarket API)
POLITICS_TAG_IDS = [
    126,  # Trump
    24,  # USA Election
    1101,  # US Election
    377,  # elections 2024
    766,  # u.s. congress
    100199,  # Senate
    871,  # vice president
]

# Politics-related keywords for text-based filtering
POLITICS_KEYWORDS = [
    "trump",
    "biden",
    "election",
    "president",
    "congress",
    "senate",
    "democrat",
    "republican",
    "vote",
    "ballot",
    "governor",
    "poll",
    "political",
    "government",
    "white house",
    "cabinet",
    "impeach",
    "campaign",
    "primary",
    "nominee",
    "electoral",
    "midterm",
]


# ============================================================================
# WORKFLOW VISUALIZATION
# ============================================================================


def print_header(title: str):
    """Print a formatted header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_step(step_num: int, title: str, description: str = ""):
    """Print a workflow step."""
    print(f"\n┌{'─' * 68}┐")
    print(f"│ STEP {step_num}: {title:<58} │")
    print(f"└{'─' * 68}┘")
    if description:
        print(f"  {description}")


def print_market_card(market: Dict[str, Any], index: int = 0):
    """Print a formatted market card."""
    question = market.get("question", "N/A")[:60]
    outcomes = market.get("outcomes", [])
    prices = market.get("outcomePrices", [])
    volume = market.get("volume", 0) or 0
    active = market.get("active", False)

    # Ensure volume is a float
    try:
        volume = float(volume) if volume else 0.0
    except (ValueError, TypeError):
        volume = 0.0

    # Parse prices if they're strings
    if isinstance(prices, str):
        try:
            prices = json.loads(prices)
        except:
            prices = []

    if isinstance(outcomes, str):
        try:
            outcomes = json.loads(outcomes)
        except:
            outcomes = []

    status = "🟢 ACTIVE" if active else "🔴 CLOSED"

    print(f"\n  ┌{'─' * 64}┐")
    print(f"  │ #{index + 1}: {question:<55} │")
    print(f"  ├{'─' * 64}┤")

    # Show prices
    if len(outcomes) >= 2 and len(prices) >= 2:
        try:
            yes_price = float(prices[0]) * 100
            no_price = float(prices[1]) * 100
            print(
                f"  │   YES: {yes_price:5.1f}%  |  NO: {no_price:5.1f}%  |  Vol: ${volume:,.0f}  │"
            )
        except:
            print(f"  │   Prices: N/A  |  Vol: ${volume:,.0f}                       │")
    else:
        print(f"  │   Vol: ${volume:,.0f}                                          │")

    print(f"  │   Status: {status:<52} │")
    print(f"  └{'─' * 64}┘")


def print_summary(total: int, active: int, stored: int):
    """Print workflow summary."""
    print("\n" + "━" * 70)
    print("  📊 WORKFLOW SUMMARY")
    print("━" * 70)
    print(f"  │ Total Markets Fetched:    {total:>5}")
    print(f"  │ Active Markets:           {active:>5}")
    print(f"  │ Markets Stored in DB:     {stored:>5}")
    print("━" * 70)


# ============================================================================
# API FUNCTIONS
# ============================================================================


def fetch_politics_events(limit: int = 20) -> List[Dict]:
    """
    Fetch political events from Gamma API.

    Strategy: Use tag-based filtering for politics-related events.
    """
    all_events = []

    for tag_id in POLITICS_TAG_IDS:
        try:
            params = {
                "tag_id": tag_id,
                "active": True,
                "closed": False,
                "limit": limit,
            }
            response = httpx.get(f"{GAMMA_API_URL}/events", params=params, timeout=30)
            if response.status_code == 200:
                events = response.json()
                all_events.extend(events)
                print(f"    ✓ Tag {tag_id}: Found {len(events)} events")
            else:
                print(f"    ✗ Tag {tag_id}: HTTP {response.status_code}")
        except Exception as e:
            print(f"    ✗ Tag {tag_id}: Error - {e}")

        time.sleep(0.2)  # Rate limiting

    # Deduplicate by event ID
    seen_ids = set()
    unique_events = []
    for event in all_events:
        eid = event.get("id")
        if eid and eid not in seen_ids:
            seen_ids.add(eid)
            unique_events.append(event)

    return unique_events


def fetch_politics_markets(limit: int = 50) -> List[Dict]:
    """
    Fetch political markets using multiple strategies:
    1. Tag-based filtering
    2. Keyword matching on questions
    """
    all_markets = []

    # Strategy 1: Fetch by tags
    print("  Strategy 1: Tag-based filtering...")
    for tag_id in POLITICS_TAG_IDS:
        try:
            params = {
                "tag_id": tag_id,
                "active": True,
                "closed": False,
                "limit": limit // len(POLITICS_TAG_IDS),
            }
            response = httpx.get(f"{GAMMA_API_URL}/markets", params=params, timeout=30)
            if response.status_code == 200:
                markets = response.json()
                all_markets.extend(markets)
                print(f"    ✓ Tag {tag_id}: Found {len(markets)} markets")
        except Exception as e:
            print(f"    ✗ Tag {tag_id}: Error - {e}")

        time.sleep(0.2)

    # Strategy 2: Fetch recent active markets and filter by keywords
    print("\n  Strategy 2: Keyword filtering from recent markets...")
    try:
        params = {
            "active": True,
            "closed": False,
            "order": "id",
            "ascending": False,
            "limit": 100,
        }
        response = httpx.get(f"{GAMMA_API_URL}/markets", params=params, timeout=30)
        if response.status_code == 200:
            recent_markets = response.json()

            # Filter by keywords
            keyword_matches = []
            for market in recent_markets:
                question = (market.get("question") or "").lower()
                desc = (market.get("description") or "").lower()
                text = f"{question} {desc}"

                if any(kw in text for kw in POLITICS_KEYWORDS):
                    keyword_matches.append(market)

            all_markets.extend(keyword_matches)
            print(f"    ✓ Keyword matches: {len(keyword_matches)} markets")
    except Exception as e:
        print(f"    ✗ Keyword search error: {e}")

    # Deduplicate by market ID
    seen_ids = set()
    unique_markets = []
    for market in all_markets:
        mid = market.get("id")
        if mid and mid not in seen_ids:
            seen_ids.add(mid)
            unique_markets.append(market)

    return unique_markets


def get_market_from_event(event: Dict) -> List[Dict]:
    """Extract markets from an event object."""
    return event.get("markets", [])


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================


def store_markets(
    markets: List[Dict], memory: MemoryManager, dry_run: bool = False
) -> int:
    """Store markets in the SQLite database."""
    stored_count = 0

    for market in markets:
        if dry_run:
            print(
                f"    [DRY RUN] Would store: {market.get('id')} - {market.get('question', '')[:40]}..."
            )
            stored_count += 1
            continue

        try:
            # Parse outcomes and prices if they're strings (API sometimes returns JSON strings)
            outcomes = market.get("outcomes", [])
            if isinstance(outcomes, str):
                try:
                    outcomes = json.loads(outcomes)
                except:
                    outcomes = []

            prices = market.get("outcomePrices", [])
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except:
                    prices = []

            # Prepare market data for storage
            market_data = {
                "id": market.get("id"),
                "question": market.get("question"),
                "description": market.get("description"),
                "outcomes": outcomes,
                "outcome_prices": prices,
                "volume": market.get("volume", 0),
                "liquidity": market.get("liquidity", 0),
                "active": market.get("active", False),
                "endDate": market.get("endDate"),
            }

            memory.add_market(market_data)
            stored_count += 1
        except Exception as e:
            print(f"    ✗ Error storing market {market.get('id')}: {e}")

    return stored_count


# ============================================================================
# MAIN WORKFLOW
# ============================================================================


def run_politics_workflow(limit: int = 20, dry_run: bool = False, verbose: bool = True):
    """
    Execute the complete politics workflow.

    Args:
        limit: Maximum markets to fetch
        dry_run: If True, don't actually store to database
        verbose: If True, show detailed output
    """
    start_time = datetime.now()

    print_header("POLYMARKET POLITICS WORKFLOW")
    print(f"  Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Limit: {limit} markets")
    print(f"  Mode: {'DRY RUN' if dry_run else 'LIVE'}")

    # ── STEP 1: Fetch Markets ──
    print_step(
        1,
        "FETCH POLITICAL MARKETS",
        "Querying Gamma API with politics tags and keywords...",
    )

    markets = fetch_politics_markets(limit=limit)
    print(f"\n  📥 Fetched {len(markets)} unique political markets")

    # ── STEP 2: Filter Active Markets ──
    print_step(2, "FILTER ACTIVE MARKETS", "Removing closed/archived markets...")

    active_markets = [m for m in markets if m.get("active") and not m.get("closed")]
    print(f"\n  ✅ {len(active_markets)} active, tradeable markets")

    # ── STEP 3: Display Markets ──
    print_step(3, "PREVIEW MARKETS", "Showing top markets by relevance...")

    display_count = min(5, len(active_markets))
    for i, market in enumerate(active_markets[:display_count]):
        print_market_card(market, i)

    if len(active_markets) > display_count:
        print(f"\n  ... and {len(active_markets) - display_count} more markets")

    # ── STEP 4: Store in Database ──
    print_step(4, "STORE IN DATABASE", f"Saving to SQLite database (data/memory.db)...")

    if not dry_run:
        memory = MemoryManager(db_path="data/memory.db")
        stored_count = store_markets(active_markets, memory, dry_run=dry_run)
    else:
        stored_count = len(active_markets)
        print(f"\n  [DRY RUN] Would store {stored_count} markets")

    # ── SUMMARY ──
    print_summary(total=len(markets), active=len(active_markets), stored=stored_count)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    print(f"\n  ⏱️  Workflow completed in {duration:.2f} seconds")

    # Return results for programmatic use
    return {
        "total_fetched": len(markets),
        "active_markets": len(active_markets),
        "stored_count": stored_count,
        "markets": active_markets,
        "duration_seconds": duration,
    }


def view_stored_markets(limit: int = 10):
    """View markets stored in the database."""
    print_header("STORED POLITICAL MARKETS")

    memory = MemoryManager(db_path="data/memory.db")
    markets = memory.list_recent_markets(limit=limit)

    if not markets:
        print("\n  No markets found in database.")
        return

    print(f"\n  Found {len(markets)} markets in database:\n")

    for i, market in enumerate(markets):
        question = market.get("question", "N/A")[:55]
        m_id = market.get("id")
        active = "🟢" if market.get("active") else "🔴"
        updated = market.get("last_updated", "")[:19]

        print(f"  {active} [{m_id}] {question}")
        print(f"     Updated: {updated}")
        print()


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Polymarket Politics Workflow - Fetch and store political prediction markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/python/politics_workflow.py              # Run with defaults
  python scripts/python/politics_workflow.py --limit 50   # Fetch more markets
  python scripts/python/politics_workflow.py --dry-run    # Preview without storing
  python scripts/python/politics_workflow.py --view       # View stored markets
        """,
    )

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=20,
        help="Maximum markets to fetch (default: 20)",
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview workflow without storing to database",
    )

    parser.add_argument(
        "--view", "-v", action="store_true", help="View markets stored in database"
    )

    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output")

    args = parser.parse_args()

    if args.view:
        view_stored_markets(limit=args.limit)
    else:
        run_politics_workflow(
            limit=args.limit, dry_run=args.dry_run, verbose=not args.quiet
        )


if __name__ == "__main__":
    main()

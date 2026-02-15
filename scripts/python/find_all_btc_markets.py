"""
Comprehensive Bitcoin market search on Polymarket.
"""

import requests
from typing import List, Dict
from datetime import datetime


def fetch_markets(
    search: str, active: bool = None, closed: bool = None, limit: int = 200
) -> List[Dict]:
    """Query Gamma API."""
    url = "https://gamma-api.polymarket.com/markets"
    params = {"limit": limit}

    if search:
        params["search"] = search
    if active is not None:
        params["active"] = str(active).lower()
    if closed is not None:
        params["closed"] = str(closed).lower()

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def main():
    print("\n" + "=" * 70)
    print("🔍 COMPREHENSIVE BITCOIN MARKET SEARCH")
    print("=" * 70 + "\n")

    # Search all Bitcoin markets (active and closed)
    print("📡 Fetching all Bitcoin markets...")
    all_btc = fetch_markets(search="Bitcoin", active=None, closed=None, limit=200)

    # Filter to only markets with "bitcoin" or "btc" in question
    btc_keywords = ["bitcoin", "btc"]
    all_btc = [
        m
        for m in all_btc
        if any(kw in m.get("question", "").lower() for kw in btc_keywords)
    ]

    print(f"✅ Found {len(all_btc)} Bitcoin markets\n")

    # Categorize by type
    active_markets = [m for m in all_btc if m.get("active", False)]
    closed_markets = [m for m in all_btc if m.get("closed", False)]

    # Look for 15-minute markers
    keywords = ["15m", "15 min", "15-min", "fifteen min", "up or down"]
    short_term = []

    for market in all_btc:
        question = market.get("question", "").lower()
        if any(kw in question for kw in keywords):
            short_term.append(market)

    print(f"📊 Active: {len(active_markets)}")
    print(f"📊 Closed: {len(closed_markets)}")
    print(f"📊 Short-term/15-min candidates: {len(short_term)}\n")

    # Show active markets
    if active_markets:
        print("=" * 70)
        print("🟢 ACTIVE BITCOIN MARKETS")
        print("=" * 70 + "\n")
        for i, m in enumerate(active_markets[:10], 1):
            volume = float(m.get("volume", 0)) if m.get("volume") else 0
            print(f"{i}. ID: {m['id']}")
            print(f"   Q: {m['question']}")
            print(f"   Volume: ${volume:,.2f}")
            print(f"   Ends: {m.get('end_date_iso', 'N/A')[:10]}\n")

    # Show short-term markets (active or recently closed)
    if short_term:
        print("=" * 70)
        print("⏱️  SHORT-TERM/15-MIN MARKETS")
        print("=" * 70 + "\n")
        for i, m in enumerate(short_term[:10], 1):
            status = "🟢 ACTIVE" if m.get("active") else "🔴 CLOSED"
            print(f"{i}. {status} | ID: {m['id']}")
            print(f"   Q: {m['question']}")
            print(f"   Ends: {m.get('end_date_iso', 'N/A')}\n")

    # Show most liquid active markets
    print("=" * 70)
    print("💰 TOP 5 BY VOLUME (Active)")
    print("=" * 70 + "\n")

    sorted_by_volume = sorted(
        [m for m in active_markets],
        key=lambda x: float(x.get("volume", 0)) if x.get("volume") else 0,
        reverse=True,
    )

    for i, m in enumerate(sorted_by_volume[:5], 1):
        volume = float(m.get("volume", 0)) if m.get("volume") else 0
        print(f"{i}. ID: {m['id']}")
        print(f"   Q: {m['question'][:60]}...")
        print(f"   Volume: ${volume:,.2f}\n")

    # Summary for code use
    print("=" * 70)
    print("📝 MARKET IDS FOR YOUR CODE")
    print("=" * 70 + "\n")

    if sorted_by_volume:
        print("# Top liquid Bitcoin markets:")
        for m in sorted_by_volume[:3]:
            print(f"MARKET_ID = '{m['id']}'  # {m['question'][:50]}")

    if short_term:
        print("\n# Short-term markets (if any active):")
        for m in [x for x in short_term if x.get("active")][:3]:
            print(f"MARKET_ID = '{m['id']}'  # {m['question'][:50]}")

    # Check for "Up or Down" specifically
    updown = [m for m in all_btc if "up or down" in m.get("question", "").lower()]
    if updown:
        print(f"\n# 'Up or Down' markets found: {len(updown)}")
        for m in updown[:3]:
            status = "ACTIVE" if m.get("active") else "CLOSED"
            print(f"MARKET_ID = '{m['id']}'  # [{status}] {m['question'][:50]}")


if __name__ == "__main__":
    main()

"""
Find Bitcoin Up or Down prediction markets on Polymarket.
These are typically short-duration markets (15min, 1hr, etc.).
"""

import requests
from typing import List, Dict
from datetime import datetime, timedelta


def search_markets(query: str, limit: int = 100) -> List[Dict]:
    """Search Gamma API with query."""
    url = "https://gamma-api.polymarket.com/markets"
    params = {"limit": limit, "search": query}

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error: {e}")
        return []


def get_market_by_id(market_id: str) -> Dict:
    """Get specific market by ID."""
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"❌ Error fetching market {market_id}: {e}")
        return {}


def main():
    print("\n" + "=" * 70)
    print("🔍 SEARCHING FOR UP OR DOWN MARKETS")
    print("=" * 70 + "\n")

    # Try different search terms
    search_terms = [
        "up or down",
        "updown",
        "Bitcoin price",
        "BTC price prediction",
        "Will Bitcoin",
        "crypto price",
    ]

    all_markets = {}

    for term in search_terms:
        print(f"🔎 Searching: '{term}'")
        markets = search_markets(term, limit=50)

        for m in markets:
            market_id = m.get("id")
            if market_id and market_id not in all_markets:
                all_markets[market_id] = m

        print(f"   Found {len(markets)} markets")

    print(f"\n📊 Total unique markets: {len(all_markets)}\n")

    # Filter for recent/active markets
    now = datetime.utcnow()
    recent_cutoff = now - timedelta(days=30)

    active = []
    recent = []
    updown = []

    for m in all_markets.values():
        # Check if "up or down" in question
        question = m.get("question", "").lower()
        if "up or down" in question or "updown" in question:
            updown.append(m)

        # Check if active
        if m.get("active"):
            active.append(m)

        # Check if recent (has end_date_iso)
        end_date = m.get("end_date_iso")
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                if end_dt > recent_cutoff:
                    recent.append(m)
            except:
                pass

    print(f"🟢 Active markets: {len(active)}")
    print(f"📅 Recent markets (last 30 days): {len(recent)}")
    print(f"🎯 'Up or Down' markets: {len(updown)}\n")

    # Show Up or Down markets
    if updown:
        print("=" * 70)
        print("🎯 UP OR DOWN MARKETS")
        print("=" * 70 + "\n")
        for i, m in enumerate(updown[:10], 1):
            status = "🟢 ACTIVE" if m.get("active") else "🔴 CLOSED"
            volume = float(m.get("volume", 0)) if m.get("volume") else 0
            print(f"{i}. {status} | ID: {m['id']}")
            print(f"   Q: {m['question']}")
            print(f"   Volume: ${volume:,.2f}")
            print(f"   Ends: {m.get('end_date_iso', 'N/A')}\n")

    # Show active markets
    if active:
        print("=" * 70)
        print("🟢 ACTIVE MARKETS")
        print("=" * 70 + "\n")
        for i, m in enumerate(active[:10], 1):
            volume = float(m.get("volume", 0)) if m.get("volume") else 0
            print(f"{i}. ID: {m['id']}")
            print(f"   Q: {m['question'][:70]}...")
            print(f"   Volume: ${volume:,.2f}")
            print(f"   Ends: {m.get('end_date_iso', 'N/A')[:10]}\n")

    # Check the specific market ID from your codebase
    print("=" * 70)
    print("🔍 CHECKING MARKET ID 574073 (from your code)")
    print("=" * 70 + "\n")

    market_574073 = get_market_by_id("574073")
    if market_574073:
        print(f"✅ Found market 574073:")
        print(f"   Q: {market_574073.get('question', 'N/A')}")
        print(f"   Active: {market_574073.get('active', False)}")
        print(f"   Closed: {market_574073.get('closed', False)}")
        volume = (
            float(market_574073.get("volume", 0)) if market_574073.get("volume") else 0
        )
        print(f"   Volume: ${volume:,.2f}")
        print(f"   End Date: {market_574073.get('end_date_iso', 'N/A')}")

    # Try other known long-term Bitcoin markets
    print("\n" + "=" * 70)
    print("🔍 CHECKING OTHER COMMON BTC MARKET IDS")
    print("=" * 70 + "\n")

    # These are some common patterns for Polymarket IDs
    test_ids = ["574073", "100000", "200000", "500000", "1000000"]

    for test_id in test_ids:
        market = get_market_by_id(test_id)
        if market and "question" in market:
            btc_related = any(
                kw in market.get("question", "").lower() for kw in ["bitcoin", "btc"]
            )
            if btc_related:
                print(f"ID {test_id}: {market['question'][:60]}...")

    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70 + "\n")

    if updown:
        print("✅ Found 'Up or Down' markets. Use these IDs:")
        for m in updown[:3]:
            print(f"   '{m['id']}'  # {m['question'][:50]}")
    else:
        print("❌ No 'Up or Down' markets found.")
        print("\n💡 These markets may be:")
        print("   1. Created on-demand and expire quickly")
        print("   2. Available only through the web UI")
        print("   3. Migrated to a different API endpoint")
        print("\n🔗 Next steps:")
        print("   1. Check Polymarket.com/events directly")
        print("   2. Monitor the API for new markets")
        print("   3. Use existing Bitcoin price prediction markets")

    if active:
        print(f"\n✅ Found {len(active)} active markets you can use instead:")
        for m in sorted(
            active,
            key=lambda x: float(x.get("volume", 0)) if x.get("volume") else 0,
            reverse=True,
        )[:3]:
            print(f"   '{m['id']}'  # {m['question'][:50]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Quick betting script specifically for Up/Down markets.

Usage:
    python scripts/quick_bet_updown.py <market_id> <UP|DOWN> <amount_usd>

Example:
    python scripts/quick_bet_updown.py 1163212 UP 10.0
    (Bet $10 that Bitcoin will be UP in the time window)
"""

import sys
import requests
from polymarket_agents.connectors.polymarket import Polymarket


def get_market_from_gamma(market_id):
    """Fetch market details from Gamma API."""
    url = f"https://gamma-api.polymarket.com/markets/{market_id}"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching market: {e}")
        return None


def main():
    if len(sys.argv) < 4:
        print(
            "Usage: python scripts/quick_bet_updown.py <market_id> <UP|DOWN> <amount_usd>"
        )
        print("\nExample: python scripts/quick_bet_updown.py 1163212 UP 10.0")
        print("         (Bet $10 that price will be UP)")
        sys.exit(1)

    market_id = sys.argv[1]
    direction = sys.argv[2].upper()
    amount_usd = float(sys.argv[3])

    if direction not in ["UP", "DOWN"]:
        print("❌ Direction must be 'UP' or 'DOWN'")
        sys.exit(1)

    print(
        f"\n🎲 Preparing to bet ${amount_usd:.2f} {direction} on market {market_id}...\n"
    )

    # Fetch market details
    print("📊 Fetching market details from Gamma API...")
    market = get_market_from_gamma(market_id)

    if not market:
        print("❌ Market not found or API error")
        sys.exit(1)

    print("\n" + "=" * 70)
    print(f"Market: {market.get('question', 'N/A')}")
    print(f"Description: {market.get('description', 'N/A')[:100]}...")
    print(f"End Date: {market.get('end_date_iso', 'N/A')}")
    print("=" * 70)

    # Get token IDs
    tokens = market.get("tokens", [])
    if not tokens or len(tokens) < 2:
        print("❌ Market doesn't have proper token structure")
        sys.exit(1)

    # Typically: index 0 = UP, index 1 = DOWN (but verify with outcome names)
    up_token = None
    down_token = None

    for token in tokens:
        outcome = token.get("outcome", "").upper()
        if "UP" in outcome or "YES" in outcome or "HIGHER" in outcome:
            up_token = token.get("token_id")
        elif "DOWN" in outcome or "NO" in outcome or "LOWER" in outcome:
            down_token = token.get("token_id")

    # Fallback to index-based if outcome names unclear
    if not up_token and len(tokens) >= 1:
        up_token = tokens[0].get("token_id")
    if not down_token and len(tokens) >= 2:
        down_token = tokens[1].get("token_id")

    token_id = up_token if direction == "UP" else down_token

    if not token_id:
        print(f"❌ Could not find token ID for {direction}")
        sys.exit(1)

    print(f"\n📊 Betting Details:")
    print(f"   Direction: {direction}")
    print(f"   Token ID: {token_id}")
    print(f"   Amount: ${amount_usd:.2f} USDC")

    # Check balance
    poly = Polymarket()
    if not poly.client:
        print("\n❌ Polymarket client not initialized")
        print("   Check POLYGON_WALLET_PRIVATE_KEY in .env")
        sys.exit(1)

    balance = poly.get_usdc_balance()
    print(f"\n💰 Wallet Balance: ${balance:.2f} USDC")

    if balance < amount_usd:
        print(
            f"\n❌ Insufficient funds! You need ${amount_usd:.2f} but only have ${balance:.2f}"
        )
        sys.exit(1)

    # Confirm
    confirm = (
        input(f"\n⚠️  Execute BET ${amount_usd:.2f} {direction}? (yes/no): ")
        .strip()
        .lower()
    )

    if confirm not in ["yes", "y"]:
        print("❌ Bet cancelled.")
        sys.exit(0)

    print("\n🚀 Placing MARKET order (instant execution)...")

    try:
        from polymarket_agents.tools.trade_tools import _execute_market_order_impl

        result = _execute_market_order_impl(
            token_id=token_id, amount=amount_usd, side="BUY"
        )

        print("\n✅ Order executed!")
        print(result)

        print(f"\n⏰ Market expires: {market.get('end_date_iso', 'N/A')}")
        print("💡 Monitor your position:")
        print(
            "   python -c 'from polymarket_agents.langchain.clob_tools import clob_get_open_orders; print(clob_get_open_orders())'"
        )

    except Exception as e:
        print(f"\n❌ Error placing bet: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

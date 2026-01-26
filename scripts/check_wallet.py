#!/usr/bin/env python3
"""Check Polymarket wallet status and balance."""

from polymarket_agents.connectors.polymarket import Polymarket


def main():
    print("\n💰 Checking wallet status...\n")

    try:
        poly = Polymarket()

        if not poly.client:
            print("❌ Polymarket client not initialized")
            print("   Check POLYGON_WALLET_PRIVATE_KEY in .env")
            return

        address = poly.get_address_for_private_key()
        balance = poly.get_usdc_balance()

        print("=" * 70)
        print(f"✅ Wallet Address: {address}")
        print(f"✅ USDC Balance:   ${balance:.2f}")
        print("=" * 70)

        if balance == 0:
            print("\n⚠️  Your wallet has no funds!")
            print("\n📝 To fund your wallet:")
            print(f"   1. Send USDC (Polygon network) to: {address}")
            print("   2. Minimum: $10 USDC recommended for testing")
            print("   3. Bridge from Ethereum: https://wallet.polygon.technology/")
            print("   4. Or buy USDC directly on Polygon via an exchange")
            print("\n🔗 Quick links:")
            print(
                f"   - View on PolygonScan: https://polygonscan.com/address/{address}"
            )
            print("   - Add funds: https://wallet.polygon.technology/")
        else:
            print(f"\n✅ You have ${balance:.2f} USDC available for trading!")
            print("\n💡 Next steps:")
            print("   1. Find markets: python scripts/find_bitcoin_markets.py")
            print(
                '   2. Analyze a market: python -m polymarket_agents.graph.planning_agent "Should I bet on Bitcoin?"'
            )

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure:")
        print("  - POLYGON_WALLET_PRIVATE_KEY is set in .env")
        print("  - The private key is a valid 64-character hex string")


if __name__ == "__main__":
    main()

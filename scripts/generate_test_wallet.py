#!/usr/bin/env python3
"""
Generate a new Ethereum wallet for Polymarket trading.

⚠️  WARNING: For testing only! Store the private key securely.
⚠️  Never commit private keys to git or share them publicly.
"""

from eth_account import Account
import secrets
import os


def generate_wallet():
    """Generate a new Ethereum wallet with private key."""

    # Generate cryptographically secure random private key
    private_key = secrets.token_hex(32)  # 64 hex characters

    # Create account from private key
    account = Account.from_key(private_key)

    return {
        "private_key": private_key,
        "address": account.address,
    }


def main():
    print("=" * 70)
    print("POLYMARKET TRADING WALLET GENERATOR")
    print("=" * 70)
    print("\n⚠️  SECURITY WARNING:")
    print("   - This generates a NEW wallet for testing")
    print("   - SAVE the private key in a secure location")
    print("   - NEVER share your private key with anyone")
    print("   - Start with small amounts ($5-10) for testing")
    print("\n" + "=" * 70)

    response = input("\nGenerate new wallet? (yes/no): ").strip().lower()

    if response not in ["yes", "y"]:
        print("Cancelled.")
        return

    print("\n🔐 Generating new wallet...\n")

    wallet = generate_wallet()

    print("✅ Wallet Generated Successfully!\n")
    print("📋 WALLET DETAILS:")
    print("─" * 70)
    print(f"Private Key: {wallet['private_key']}")
    print(f"Address:     {wallet['address']}")
    print("─" * 70)

    print("\n📝 NEXT STEPS:")
    print("\n1. Update your .env file:")
    print(f"   POLYGON_WALLET_PRIVATE_KEY={wallet['private_key']}")

    print("\n2. Fund your wallet:")
    print(f"   - Send USDC (Polygon network) to: {wallet['address']}")
    print("   - Minimum: $10 USDC recommended for testing")
    print("   - Bridge from Ethereum: https://wallet.polygon.technology/")

    print("\n3. Verify setup:")
    print(
        '   python -c "from polymarket_agents.connectors.polymarket import Polymarket;'
    )
    print("   p=Polymarket(); print(f'Balance: ${p.get_usdc_balance():.2f}')\"")

    print("\n4. Start trading:")
    print('   python -m polymarket_agents.graph.planning_agent "Find Bitcoin markets"')

    print("\n⚠️  IMPORTANT:")
    print("   - Save the private key in a password manager or secure note")
    print("   - This wallet is for the Polygon network (not Ethereum mainnet)")
    print("   - Never commit the .env file to git")
    print("   - For production, consider using hardware wallets")

    print("\n" + "=" * 70)

    # Offer to update .env automatically
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        update = (
            input("\n📝 Update .env file automatically? (yes/no): ").strip().lower()
        )
        if update in ["yes", "y"]:
            try:
                # Read existing .env
                with open(env_path, "r") as f:
                    lines = f.readlines()

                # Update or add POLYGON_WALLET_PRIVATE_KEY
                updated = False
                for i, line in enumerate(lines):
                    if line.startswith("POLYGON_WALLET_PRIVATE_KEY="):
                        lines[i] = (
                            f"POLYGON_WALLET_PRIVATE_KEY={wallet['private_key']}\n"
                        )
                        updated = True
                        break

                if not updated:
                    lines.append(
                        f"\n# Auto-generated wallet\nPOLYGON_WALLET_PRIVATE_KEY={wallet['private_key']}\n"
                    )

                # Write back
                with open(env_path, "w") as f:
                    f.writelines(lines)

                print("✅ .env file updated successfully!")

            except Exception as e:
                print(f"❌ Error updating .env: {e}")
                print("   Please update manually.")


if __name__ == "__main__":
    main()

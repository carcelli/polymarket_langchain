"""
Example: Track 15-minute Bitcoin Up or Down markets on Polymarket.

This demonstrates how to:
1. Find the current 15-minute market
2. Fetch live market data
3. Monitor price changes in real-time
4. Switch to next market when current one expires

Usage:
    python examples/track_15min_bitcoin.py
    python examples/track_15min_bitcoin.py --monitor  # Run continuously
"""

import argparse
import time
from datetime import datetime, timezone
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from polymarket_agents.connectors.updown_markets import UpDownMarketConnector


def display_market_info(market: dict):
    """Display formatted market information."""
    print("\n" + "=" * 70)
    print(f"📊 {market['question']}")
    print("=" * 70)
    print(f"Market ID: {market['market_id']}")
    print(f"Slug: {market['slug']}")
    print(f"End Date: {market['end_date']}")
    print(f"\n💰 Market Stats:")
    print(f"   Volume: ${market['volume']:,.2f}")
    print(f"   Liquidity: ${market['liquidity']:,.2f}")
    print(f"\n📈 Outcome Probabilities:")
    print(f"   Up:   {market['up_price']:.3f} ({market['up_price']*100:.1f}%)")
    print(f"   Down: {market['down_price']:.3f} ({market['down_price']*100:.1f}%)")
    print(f"\n🔄 Status:")
    print(f"   Active: {market['active']}")
    print(f"   Closed: {market['closed']}")


def monitor_market(connector: UpDownMarketConnector, duration_minutes: int = 15):
    """
    Monitor a 15-minute market in real-time.

    Args:
        connector: UpDownMarketConnector instance
        duration_minutes: How long to monitor (default: 15 min)
    """
    market = connector.get_current_market_data()

    if not market:
        print("❌ Could not find active market")
        return

    display_market_info(market)

    print(f"\n🔔 Monitoring started at {datetime.now().strftime('%H:%M:%S')}")
    print("   Press Ctrl+C to stop\n")

    market_id = market["market_id"]
    start_time = time.time()
    poll_count = 0

    try:
        while time.time() - start_time < duration_minutes * 60:
            poll_count += 1

            # Fetch latest data
            updated = connector.fetch_market_data(market_id)

            if not updated or not updated.get("active"):
                print("\n⏰ Market has closed or expired!")
                print("   Outcome:", updated.get("outcome", "Pending resolution"))

                # Switch to next market
                print("\n🔄 Switching to next 15-minute interval...")
                new_market = connector.get_current_market_data()

                if new_market:
                    market_id = new_market["market_id"]
                    display_market_info(new_market)
                else:
                    print("❌ No new market found, ending monitoring")
                    break
            else:
                # Parse updated prices
                import json

                outcome_prices = json.loads(
                    updated.get("outcomePrices", '["0.5", "0.5"]')
                )
                up_price = float(outcome_prices[0])
                down_price = float(outcome_prices[1])
                volume = float(updated.get("volume", 0))

                # Display update
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] Up: {up_price:.3f} | Down: {down_price:.3f} | Vol: ${volume:,.0f}"
                )

            # Poll every 30 seconds
            time.sleep(30)

    except KeyboardInterrupt:
        print(f"\n\n⏹️  Monitoring stopped after {poll_count} polls")
        print(f"   Duration: {(time.time() - start_time)/60:.1f} minutes")


def show_upcoming_markets(connector: UpDownMarketConnector):
    """Display upcoming 15-minute market slots."""
    print("\n" + "=" * 70)
    print("📅 UPCOMING 15-MINUTE MARKETS")
    print("=" * 70 + "\n")

    upcoming = connector.get_upcoming_markets(num_intervals=5)

    for i, market in enumerate(upcoming, 1):
        dt = datetime.fromisoformat(market["datetime"])
        print(f"{i}. {dt.strftime('%H:%M:%S UTC')} ({dt.strftime('%I:%M %p ET')})")
        print(f"   Timestamp: {market['timestamp']}")
        print(f"   URL: {market['url']}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Track 15-minute Bitcoin Up or Down markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor markets continuously in real-time",
    )
    parser.add_argument(
        "--upcoming", action="store_true", help="Show upcoming market time slots"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=15,
        help="Monitoring duration in minutes (default: 15)",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🎯 15-MINUTE BITCOIN MARKET TRACKER")
    print("=" * 70)

    connector = UpDownMarketConnector()

    # Show current time slot
    current_slot = connector.get_current_15min_slot()
    next_slot = connector.get_next_15min_slot()

    print(
        f"\n⏰ Current Slot: {current_slot.strftime('%H:%M UTC')} ({current_slot.strftime('%I:%M %p ET')})"
    )
    print(
        f"⏰ Next Slot: {next_slot.strftime('%H:%M UTC')} ({next_slot.strftime('%I:%M %p ET')})"
    )

    if args.upcoming:
        show_upcoming_markets(connector)

    elif args.monitor:
        monitor_market(connector, duration_minutes=args.duration)

    else:
        # Default: show current market snapshot
        market = connector.get_current_market_data()

        if market:
            display_market_info(market)

            print("\n" + "=" * 70)
            print("💡 Next Steps:")
            print("=" * 70)
            print("\n1. Monitor in real-time:")
            print("   python examples/track_15min_bitcoin.py --monitor")
            print("\n2. View upcoming markets:")
            print("   python examples/track_15min_bitcoin.py --upcoming")
            print("\n3. Monitor for custom duration:")
            print("   python examples/track_15min_bitcoin.py --monitor --duration 30")
        else:
            print("\n❌ Could not fetch current market data")
            print("\n💡 Troubleshooting:")
            print("   - Check internet connection")
            print("   - Verify Polymarket is accessible")
            print("   - Try again in a few seconds")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Gamma Markets CLI Tool

Command-line interface for testing Polymarket Gamma API integration.
Provides quick validation of market discovery and probability extraction.

Usage:
    python -m scripts.cli.gamma_cli --keyword "recession" --limit 10
    python -m scripts.cli.gamma_cli --active-only --limit 5
"""

import argparse
import json
import sys
from typing import Optional

# Add src to path for imports
sys.path.insert(0, "src")

from polymarket_agents.tools.gamma_markets import GammaMarketsTool


def format_market_summary(market: dict) -> str:
    """Format a market snapshot for CLI display."""
    question = market.get("question", "N/A")[:80] + (
        "..." if len(market.get("question", "")) > 80 else ""
    )
    yes_prob = market.get("yes_prob", 0)
    volume = market.get("volume", 0)
    liquidity = market.get("liquidity", 0)

    return (
        f"📊 {question}\n"
        f"   Probability: {yes_prob:.1%} YES / {(1-yes_prob):.1%} NO\n"
        f"   Volume: ${volume:,.0f} | Liquidity: {liquidity:.2f}\n"
        f"   Slug: {market.get('slug', 'N/A')}\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test Polymarket Gamma API integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.cli.gamma_cli --limit 5
  python -m scripts.cli.gamma_cli --keyword "election" --limit 10
  python -m scripts.cli.gamma_cli --active-only --limit 20 --json
        """,
    )

    parser.add_argument(
        "--keyword", "-k", type=str, help="Filter markets by question keyword"
    )

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=10,
        help="Maximum number of markets to fetch (default: 10)",
    )

    parser.add_argument(
        "--active-only",
        "-a",
        action="store_true",
        help="Only fetch active/open markets",
    )

    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output raw JSON instead of formatted display",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed market information"
    )

    parser.add_argument(
        "--edge-ranking",
        "-e",
        action="store_true",
        help="Rank markets by edge score (volume × |0.5 - prob|)",
    )

    parser.add_argument(
        "--min-volume",
        "-m",
        type=float,
        default=1000,
        help="Minimum volume for edge ranking (default: 1000)",
    )

    args = parser.parse_args()

    try:
        # Initialize the tool
        tool = GammaMarketsTool()

        print("🔍 Fetching markets from Polymarket Gamma API...")
        print(f"   Active only: {args.active_only}")
        print(f"   Limit: {args.limit}")
        if args.keyword:
            print(f"   Keyword filter: '{args.keyword}'")
        print()

        # Fetch markets
        if args.edge_ranking:
            result = tool.get_markets_with_edge(
                limit=args.limit, min_volume=args.min_volume
            )
            print(
                f"   Edge ranking: volume × |0.5 - prob|, min volume: ${args.min_volume:,.0f}"
            )
        else:
            result = tool._run(
                active=args.active_only,
                limit=args.limit,
                question_contains=args.keyword,
            )

        if not result:
            print("❌ No markets found or API error")
            return 1

        # Check for errors
        if isinstance(result[0], dict) and "error" in result[0]:
            print(f"❌ API Error: {result[0]['error']}")
            return 1

        # Output results
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"✅ Found {len(result)} markets")
            print("=" * 60)

            for i, market in enumerate(result, 1):
                if args.verbose:
                    summary = format_market_summary(market)
                    if args.edge_ranking and "edge_score" in market:
                        edge_score = market["edge_score"]
                        edge = market.get("edge", 0)
                        summary += (
                            f"\n   Edge Score: {edge_score:,.0f} | Edge: {edge:.2f}"
                        )
                    print(f"{i}. {summary}")
                else:
                    question = market.get("question", "N/A")[:50] + (
                        "..." if len(market.get("question", "")) > 50 else ""
                    )
                    yes_prob = market.get("yes_prob", 0)
                    volume = market.get("volume", 0)

                    if args.edge_ranking and "edge_score" in market:
                        edge_score = market["edge_score"]
                        print(
                            f"{i:2d}. {question:<50} | {yes_prob:5.1%} | ${volume:>8,.0f} | Edge: {edge_score:>8,.0f}"
                        )
                    else:
                        print(
                            f"{i:2d}. {question:<50} | {yes_prob:5.1%} | ${volume:>8,.0f}"
                        )

                if not args.verbose:
                    print()

        # Performance metrics
        print("\n📈 Performance Summary:")
        if result:
            total_volume = sum(m.get("volume", 0) for m in result)
            avg_probability = sum(m.get("yes_prob", 0.5) for m in result) / len(result)
            print(f"   Total markets: {len(result)}")
            print(f"   Combined volume: ${total_volume:,.0f}")
            print(f"   Average probability: {avg_probability:.1%}")

        return 0

    except KeyboardInterrupt:
        print("\n⏹️  Cancelled by user")
        return 130
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

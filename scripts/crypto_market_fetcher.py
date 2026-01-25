#!/usr/bin/env python3
"""
Fetch active crypto markets from Polymarket using official Tags API.

Uses Polymarket's Gamma API with tag filtering (tag_id=21 for Crypto).
Focuses on Up/Down markets for ultra-short-term trading.
"""

import httpx
import json
import structlog
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional

from polymarket_agents.connectors.gamma import GammaMarketClient
from polymarket_agents.utils.objects import Tag, PolymarketEvent, Market

# Structured logging
logger = structlog.get_logger()

# Official Polymarket Crypto tag
CRYPTO_TAG_ID = "21"
GAMMA_API_BASE = "https://gamma-api.polymarket.com"


def fetch_crypto_tags(gamma_client: GammaMarketClient) -> List[Dict]:
    """
    Fetch crypto-related tags from Polymarket API.

    Per official docs: Use GET /tags to discover available tags,
    then filter markets with tag_id parameter.

    Returns:
        List of Tag dicts for crypto categories

    References:
        https://docs.polymarket.com/api-reference/tags/list-tags
    """
    try:
        all_tags = gamma_client.get_all_tags(limit=100)

        # Filter for crypto-related tags
        crypto_tags = [
            tag
            for tag in all_tags
            if tag.get("slug") in ["crypto", "bitcoin", "ethereum", "cryptocurrency"]
            or tag.get("label", "").lower() in ["crypto", "bitcoin", "ethereum"]
        ]

        logger.info(
            "fetched_crypto_tags",
            total_tags=len(all_tags),
            crypto_tags_found=len(crypto_tags),
            crypto_tag_slugs=[tag.get("slug") for tag in crypto_tags],
        )

        return crypto_tags

    except httpx.HTTPError as e:
        logger.error("tags_api_error", error=str(e))
        return []
    except Exception as e:
        logger.error("failed_to_fetch_tags", error=str(e))
        return []


def fetch_markets_by_slug(slug: str) -> Optional[Dict]:
    """
    Fetch a specific market by its slug.

    Per official docs: "Individual markets and events are best fetched
    using their unique slug identifier."

    Args:
        slug: Market slug from URL (e.g., 'bitcoin-up-or-down-15-minute')

    Returns:
        Market dict or None if not found

    Example:
        From URL: https://polymarket.com/event/bitcoin-up-or-down
        Extract slug: 'bitcoin-up-or-down'

    References:
        https://docs.polymarket.com/api-reference/events/get-event-by-slug
    """
    url = f"{GAMMA_API_BASE}/events/slug/{slug}"

    try:
        response = httpx.get(url, timeout=10)
        response.raise_for_status()
        event = response.json()

        logger.info(
            "fetched_market_by_slug",
            slug=slug,
            markets_count=len(event.get("markets", [])),
        )

        return event

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            logger.warning("market_not_found", slug=slug)
        else:
            logger.error("slug_fetch_error", slug=slug, status=e.response.status_code)
        return None
    except Exception as e:
        logger.error("failed_to_fetch_by_slug", slug=slug, error=str(e))
        return None


def fetch_crypto_updown_markets(
    min_volume: float = 0,
    max_duration_minutes: int = 60,
    include_expired: bool = False,
    use_tag_filter: bool = True,
    related_tags: bool = False,
) -> List[Dict]:
    """
    Fetch active crypto Up/Down markets using official Polymarket Tags API.

    Follows official Gamma API best practices:
    - Uses /events endpoint with tag_id for efficient filtering
    - Orders by ID descending (newest first) for consistency
    - Implements proper pagination with limit/offset
    - Events contain nested markets (work backwards pattern)

    Args:
        min_volume: Minimum volume in USD (default: no filter)
        max_duration_minutes: Max time until expiry (default: 60 min)
        include_expired: Include markets that already ended
        use_tag_filter: Use official tag_id=21 filter (recommended per docs)
        related_tags: Include markets with related tags

    Returns:
        List of market dicts with standardized fields

    References:
        https://docs.polymarket.com/api-reference/events/get-events
        https://docs.polymarket.com/quickstart/fetching-data
    """

    gamma_client = GammaMarketClient()

    # Build query params following official API best practices
    if use_tag_filter:
        # Use official crypto tag endpoint (recommended approach)
        # Per docs: "Most efficient for retrieving all active markets"
        url = f"{GAMMA_API_BASE}/events"
        params = {
            "tag_id": CRYPTO_TAG_ID,
            "closed": "false" if not include_expired else "true",
            "order": "id",  # Order by event ID (best practice)
            "ascending": "false",  # Newest first (best practice)
            "limit": 500,  # Max results per request
            "offset": 0,  # Pagination starting point
        }

        # Optional: Include related tags (e.g., Bitcoin, Ethereum subcategories)
        if related_tags:
            params["related_tags"] = "true"

    else:
        # Fallback: fetch all markets and filter manually
        url = f"{GAMMA_API_BASE}/markets"
        params = {
            "closed": "false" if not include_expired else "true",
            "order": "id",
            "ascending": "false",
            "limit": 500,
            "offset": 0,
        }

    try:
        response = httpx.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Handle events vs markets response structure
        # Per docs: "Events contain their associated markets"
        if use_tag_filter:
            # Events endpoint returns events with nested markets
            events = data if isinstance(data, list) else data.get("data", [])
            all_markets = []
            for event in events:
                markets = event.get("markets", [])
                # Each event may have multiple markets
                all_markets.extend(markets)
        else:
            all_markets = data if isinstance(data, list) else data.get("data", [])

        logger.info(
            "fetched_markets_from_api",
            url=url,
            use_tag_filter=use_tag_filter,
            related_tags=related_tags,
            total_markets=len(all_markets),
            api_method="events_with_tags" if use_tag_filter else "markets_direct",
        )

    except httpx.HTTPError as e:
        logger.error("api_request_failed", error=str(e), url=url)
        return []
    except Exception as e:
        logger.error("unexpected_error_fetching_markets", error=str(e))
        return []

    crypto_markets = []
    now = datetime.now(timezone.utc)

    for market in all_markets:
        question = market.get("question", "").lower()

        # Filter for Up/Down crypto markets
        is_updown = "up or down" in question

        # If not using tag filter, verify it's crypto
        if not use_tag_filter:
            is_crypto = any(
                coin in question
                for coin in [
                    "bitcoin",
                    "btc",
                    "ethereum",
                    "eth",
                    "solana",
                    "sol",
                    "xrp",
                    "doge",
                ]
            )
            if not (is_updown and is_crypto):
                continue
        elif not is_updown:
            # Using tag filter, only check for Up/Down
            continue

        # Parse end time with proper error handling
        end_date = (
            market.get("end_date_iso")
            or market.get("endDate")
            or market.get("end_date")
        )

        if not end_date:
            logger.debug("market_missing_end_date", market_id=market.get("id"))
            continue

        try:
            # Handle various datetime formats
            if isinstance(end_date, str):
                if end_date.endswith("Z"):
                    end_time = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                else:
                    end_time = datetime.fromisoformat(end_date)
                    if end_time.tzinfo is None:
                        end_time = end_time.replace(tzinfo=timezone.utc)
            else:
                continue

        except (ValueError, TypeError) as e:
            logger.warning(
                "invalid_date_format",
                market_id=market.get("id"),
                end_date=end_date,
                error=str(e),
            )
            continue

        # Calculate minutes until expiry
        time_until_end = end_time - now
        minutes_until_end = time_until_end.total_seconds() / 60

        # Skip if already expired (unless include_expired=True)
        if minutes_until_end < 0 and not include_expired:
            continue

        # Skip if too far in future
        if minutes_until_end > max_duration_minutes:
            continue

        # Extract volume
        volume = float(market.get("volume", 0))

        if volume < min_volume:
            continue

        # Parse asset from question
        asset = "UNKNOWN"
        if "bitcoin" in question or "btc" in question:
            asset = "BTC"
        elif "ethereum" in question or "eth" in question:
            asset = "ETH"
        elif "solana" in question or "sol" in question:
            asset = "SOL"
        elif "xrp" in question:
            asset = "XRP"
        elif "doge" in question:
            asset = "DOGE"

        # Parse outcomes and prices
        outcomes = market.get("outcomes", [])
        outcome_prices = market.get("outcomePrices", [])

        if isinstance(outcomes, str):
            try:
                outcomes = json.loads(outcomes.replace("'", '"'))
            except:
                outcomes = []

        if isinstance(outcome_prices, str):
            try:
                outcome_prices = [
                    float(p.strip("'\"")) for p in outcome_prices.strip("[]").split(",")
                ]
            except:
                outcome_prices = []

        # Standardize outcome format
        up_price = None
        down_price = None

        if len(outcomes) >= 2 and len(outcome_prices) >= 2:
            for i, outcome in enumerate(outcomes):
                if "up" in str(outcome).lower():
                    up_price = float(outcome_prices[i])
                elif "down" in str(outcome).lower():
                    down_price = float(outcome_prices[i])

        # If can't parse, assume first is UP, second is DOWN
        if up_price is None and len(outcome_prices) >= 2:
            up_price = float(outcome_prices[0])
            down_price = float(outcome_prices[1])

        crypto_markets.append(
            {
                "id": market.get("condition_id") or market["id"],
                "question": market["question"],
                "asset": asset,
                "category": "Crypto",
                "volume": volume,
                "end_date": end_time.isoformat(),
                "minutes_until_end": minutes_until_end,
                "duration_label": _format_duration(minutes_until_end),
                "outcomes": outcomes,
                "outcome_prices": outcome_prices,
                "up_price": up_price,
                "down_price": down_price,
                "active": market.get("active", True),
                "closed": market.get("closed", False),
                "tokens": market.get("tokens", []),
                "raw_market": market,  # Keep full data for debugging
            }
        )

    # Sort by time until end (soonest first)
    crypto_markets.sort(key=lambda x: x["minutes_until_end"])

    return crypto_markets


def _format_duration(minutes: float) -> str:
    """
    Format duration for human-readable display.

    Args:
        minutes: Duration in minutes

    Returns:
        Formatted string (e.g., "15m", "2.5h", "EXPIRED")
    """
    if minutes < 0:
        return "EXPIRED"
    elif minutes < 1:
        return f"{int(minutes * 60)}s"
    elif minutes < 60:
        return f"{int(minutes)}m"
    else:
        hours = minutes / 60
        return f"{hours:.1f}h"


def extract_start_price(market: Dict) -> Optional[float]:
    """
    Extract starting price from market question using regex.

    Example:
        "Bitcoin Up or Down, 15-minute market — price to beat $92,994.26"
        Returns: 92994.26

    Args:
        market: Market dict with 'question' field

    Returns:
        Extracted price as float, or None if not found
    """
    import re

    question = market.get("question", "")

    # Pattern: $XX,XXX.XX or $XX,XXX
    price_match = re.search(r"\$?([\d,]+\.?\d*)", question)

    if price_match:
        try:
            price_str = price_match.group(1).replace(",", "")
            return float(price_str)
        except (ValueError, AttributeError) as e:
            logger.debug("failed_to_parse_price", question=question, error=str(e))

    return None


def get_current_price(asset: str) -> Optional[float]:
    """
    Get current spot price for crypto asset from CoinGecko.

    Args:
        asset: Asset symbol (BTC, ETH, SOL, XRP, DOGE)

    Returns:
        Current USD price, or None if fetch fails

    Note:
        Uses CoinGecko public API (no auth required, rate limited).
    """

    asset_map = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "XRP": "ripple",
        "DOGE": "dogecoin",
    }

    coin_id = asset_map.get(asset)

    if not coin_id:
        logger.warning("unsupported_asset", asset=asset)
        return None

    try:
        url = "https://api.coingecko.com/api/v3/simple/price"
        params = {"ids": coin_id, "vs_currencies": "usd"}

        response = httpx.get(url, params=params, timeout=5)
        response.raise_for_status()

        data = response.json()
        price = data[coin_id]["usd"]

        logger.debug("fetched_spot_price", asset=asset, price=price)

        return price

    except httpx.HTTPError as e:
        logger.error(
            "coingecko_api_error",
            asset=asset,
            error=str(e),
            status_code=getattr(e.response, "status_code", None),
        )
        return None
    except (KeyError, ValueError) as e:
        logger.error("invalid_price_response", asset=asset, error=str(e))
        return None


def print_markets(markets: List[Dict], max_display: int = 20) -> None:
    """
    Pretty print markets to console with formatting.

    Args:
        markets: List of market dicts
        max_display: Maximum number of markets to display
    """

    if not markets:
        print("No crypto Up/Down markets found.")
        logger.info("no_markets_found")
        return

    print(f"\n{'='*80}")
    print(f"🪙 CRYPTO UP/DOWN MARKETS ({len(markets)} found)")
    print(f"{'='*80}\n")

    for i, market in enumerate(markets[:max_display]):
        print(f"[{i+1}] {market['question']}")
        print(f"    Asset: {market['asset']}")
        print(f"    Volume: ${market['volume']:,.0f}")
        print(f"    Expires: {market['duration_label']}")

        if market.get("up_price") and market.get("down_price"):
            print(
                f"    Odds: UP {market['up_price']:.0%} | DOWN {market['down_price']:.0%}"
            )

        # Try to get current price
        start_price = extract_start_price(market)
        current_price = get_current_price(market["asset"])

        if start_price and current_price:
            change = current_price - start_price
            change_pct = (change / start_price) * 100

            direction = "⬆️ UP" if change > 0 else "⬇️ DOWN"

            print(
                f"    Price: ${start_price:,.2f} → ${current_price:,.2f} ({direction} {abs(change_pct):.2f}%)"
            )

        print(f"    ID: {market['id']}")
        print()

    if len(markets) > max_display:
        print(f"... and {len(markets) - max_display} more")

    logger.info(
        "displayed_markets",
        total=len(markets),
        displayed=min(len(markets), max_display),
    )


def main() -> None:
    """
    CLI entry point for crypto market fetcher.

    Implements official Polymarket Gamma API best practices:
    - Fetch by Tags for category filtering (tag_id=21 for Crypto)
    - Uses Events endpoint for efficient market discovery
    - Supports pagination and ordering per docs
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch crypto Up/Down markets using official Polymarket Tags API",
        epilog="Official API Docs: https://docs.polymarket.com/quickstart/fetching-data",
    )
    parser.add_argument(
        "--min-volume", type=float, default=0, help="Minimum volume in USD (default: 0)"
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=60,
        help="Max minutes until expiry (default: 60)",
    )
    parser.add_argument(
        "--asset", type=str, help="Filter by asset (BTC, ETH, SOL, XRP, DOGE)"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON instead of pretty print"
    )
    parser.add_argument(
        "--no-tag-filter",
        action="store_true",
        help="Disable official tag filter (fallback to keyword matching)",
    )
    parser.add_argument(
        "--related-tags",
        action="store_true",
        help="Include markets with related crypto tags (Bitcoin, Ethereum, etc.)",
    )
    parser.add_argument(
        "--slug",
        type=str,
        help="Fetch specific market by slug (e.g., bitcoin-up-or-down-15-minute)",
    )

    args = parser.parse_args()

    # Handle slug-based fetch (official best practice for individual markets)
    if args.slug:
        logger.info("fetching_by_slug", slug=args.slug)
        event = fetch_markets_by_slug(args.slug)

        if event:
            if args.json:
                print(json.dumps(event, indent=2, default=str))
            else:
                print(f"\n📍 Event: {event.get('title', 'Unknown')}")
                print(f"Markets: {len(event.get('markets', []))}")
                for market in event.get("markets", []):
                    print(f"  - {market.get('question')}")
        else:
            print(f"Market not found: {args.slug}")
        return

    logger.info(
        "starting_crypto_market_fetch",
        min_volume=args.min_volume,
        max_duration=args.max_duration,
        asset_filter=args.asset,
        use_tag_filter=not args.no_tag_filter,
        related_tags=args.related_tags,
        method="events_endpoint_with_tags",
    )

    markets = fetch_crypto_updown_markets(
        min_volume=args.min_volume,
        max_duration_minutes=args.max_duration,
        use_tag_filter=not args.no_tag_filter,
        related_tags=args.related_tags,
    )

    # Filter by asset if specified
    if args.asset:
        original_count = len(markets)
        markets = [m for m in markets if m["asset"] == args.asset.upper()]
        logger.info(
            "filtered_by_asset",
            asset=args.asset.upper(),
            original_count=original_count,
            filtered_count=len(markets),
        )

    if args.json:
        print(json.dumps(markets, indent=2, default=str))
    else:
        print_markets(markets)

        # Summary stats
        if markets:
            total_volume = sum(m["volume"] for m in markets)
            avg_duration = sum(m["minutes_until_end"] for m in markets) / len(markets)

            by_asset: Dict[str, int] = {}
            for m in markets:
                by_asset[m["asset"]] = by_asset.get(m["asset"], 0) + 1

            print(f"\n{'='*80}")
            print("📊 SUMMARY")
            print(f"{'='*80}")
            print(f"Total markets: {len(markets)}")
            print(f"Total volume: ${total_volume:,.0f}")
            print(f"Avg duration: {avg_duration:.1f} minutes")
            print(f"\nBy asset:")
            for asset, count in sorted(by_asset.items()):
                print(f"  {asset}: {count} markets")

            logger.info(
                "summary_stats",
                total_markets=len(markets),
                total_volume=total_volume,
                avg_duration=avg_duration,
                by_asset=by_asset,
            )


if __name__ == "__main__":
    main()

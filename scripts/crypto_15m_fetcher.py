#!/usr/bin/env python3
"""
Ultra-high-velocity 15-minute crypto binary options scanner.

Fetches imminent-expiry markets from Polymarket using official Gamma API.
These are NOT gambling - they are binary options on volatility settling every 15 minutes.

Strategy:
    1. Use /events endpoint with crypto tag (official best practice)
    2. Parse endDate and filter for < 25 minutes until expiry
    3. Pagination loop to capture all active contracts
    4. Return sorted by expiry (soonest first)

Critical Notes:
    - Polymarket charges 1-3% taker fees on these markets
    - Edge calculator must demand >4-5% margin to cover vig
    - Real-time Binance/CoinGecko prices needed for arbitrage

References:
    https://docs.polymarket.com/quickstart/fetching-data
    https://docs.polymarket.com/api-reference/events/get-events
"""

import httpx
import json
import structlog
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import pandas as pd
from time import sleep

from polymarket_agents.connectors.gamma import GammaMarketClient
from polymarket_agents.utils.objects import Market, PolymarketEvent

# Structured logging
logger = structlog.get_logger()

# Configuration
GAMMA_API_BASE = "https://gamma-api.polymarket.com"
CRYPTO_TAG_ID = "21"  # Official Polymarket crypto tag
MAX_DURATION_MINUTES = 25  # Only markets expiring in < 25 min
MIN_VOLUME = 0  # Minimum volume filter (set to 0 to see all)
RATE_LIMIT_COOLDOWN = 5  # Seconds to wait on 429
REQUEST_DELAY = 0.2  # Polite delay between requests


class Crypto15MinuteFetcher:
    """
    High-velocity scanner for 15-minute crypto binary option markets.
    
    Implements official Polymarket Gamma API best practices with strict
    time-window filtering for imminent expiries.
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'PolyBot/2.0 (Crypto-15m-Scanner)',
            'Accept': 'application/json'
        }
        
    def fetch_active_15m_markets(
        self,
        max_duration_minutes: int = MAX_DURATION_MINUTES,
        min_volume: float = MIN_VOLUME,
        tag_id: str = CRYPTO_TAG_ID,
        max_pages: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Fetch ALL active crypto markets with pagination, filter for 15-minute windows.
        
        Per official docs: Uses /events endpoint (most efficient) with tag filtering.
        
        Args:
            max_duration_minutes: Maximum minutes until expiry (default: 25)
            min_volume: Minimum volume filter in USD (default: 0)
            tag_id: Polymarket tag ID for filtering (default: 21 for Crypto)
            max_pages: Maximum pages to fetch (None = all pages, set to 10-20 for fast scans)
            
        Returns:
            DataFrame with columns: expiry_min, event, question, yes_price, 
            no_price, volume, market_slug, end_date, implied_prob
            
        References:
            https://docs.polymarket.com/api-reference/events/get-events
        """
        
        all_markets: List[Dict] = []
        offset = 0
        limit = 50  # Batch size per official docs
        keep_fetching = True
        
        logger.info(
            "starting_15m_market_scan",
            tag_id=tag_id,
            max_duration=max_duration_minutes,
            min_volume=min_volume
        )
        
        while keep_fetching:
            params: Dict[str, str | int] = {
                "limit": str(limit),
                "offset": str(offset),
                "closed": "false",  # Active only (best practice)
                "tag_id": tag_id,   # Official tag filtering
                "order": "volume24hr",  # High volume first
                "ascending": "false"    # Descending order
            }
            
            try:
                # Network request with timeout
                response = httpx.get(
                    f"{GAMMA_API_BASE}/events",
                    params=params,
                    headers=self.headers,
                    timeout=10
                )
                
                # Handle rate limiting
                if response.status_code == 429:
                    logger.warning(
                        "rate_limit_hit",
                        cooldown_seconds=RATE_LIMIT_COOLDOWN
                    )
                    sleep(RATE_LIMIT_COOLDOWN)
                    continue
                
                response.raise_for_status()
                events = response.json()
                
                # Empty response = end of pagination
                if not events:
                    keep_fetching = False
                    break
                
                # Process batch
                batch_count = self._process_event_batch(
                    events=events,
                    all_markets=all_markets,
                    max_duration_minutes=max_duration_minutes,
                    min_volume=min_volume
                )
                
                logger.info(
                    "processed_batch",
                    offset=offset,
                    batch_size=len(events),
                    markets_found=batch_count
                )
                
                # Pagination
                offset += limit
                
                # Early exit if max_pages reached
                if max_pages and (offset // limit) >= max_pages:
                    logger.info("max_pages_reached", max_pages=max_pages)
                    keep_fetching = False
                    
                sleep(REQUEST_DELAY)  # Be polite to API
                
            except httpx.HTTPStatusError as e:
                logger.error(
                    "http_error_during_fetch",
                    error=str(e),
                    status_code=e.response.status_code,
                    offset=offset
                )
                keep_fetching = False
                
            except httpx.HTTPError as e:
                logger.error(
                    "http_error_during_fetch",
                    error=str(e),
                    offset=offset
                )
                keep_fetching = False
                
            except Exception as e:
                logger.error(
                    "unexpected_error_during_fetch",
                    error=str(e),
                    error_type=type(e).__name__,
                    offset=offset
                )
                keep_fetching = False
        
        # Convert to DataFrame and sort
        df = pd.DataFrame(all_markets)
        
        if not df.empty:
            df = df.sort_values(by='expiry_min', ascending=True)  # Soonest first
            
            logger.info(
                "scan_complete",
                total_markets_found=len(df),
                avg_volume=df['volume'].mean() if len(df) > 0 else 0,
                min_expiry=df['expiry_min'].min() if len(df) > 0 else 0,
                max_expiry=df['expiry_min'].max() if len(df) > 0 else 0
            )
        else:
            logger.warning("no_15m_markets_found_in_window")
        
        return df
    
    def _process_event_batch(
        self,
        events: List[Dict],
        all_markets: List[Dict],
        max_duration_minutes: int,
        min_volume: float
    ) -> int:
        """
        Process a batch of events and extract valid 15-minute markets.
        
        Args:
            events: List of event dicts from API
            all_markets: Accumulator list to append results
            max_duration_minutes: Time filter threshold
            min_volume: Volume filter threshold
            
        Returns:
            Count of valid markets found in this batch
        """
        batch_count = 0
        now = datetime.now(timezone.utc)
        
        for event in events:
            markets = event.get('markets', [])
            
            for market in markets:
                try:
                    # Parse end date (critical for time filtering)
                    end_date_str = market.get('endDate')
                    if not end_date_str:
                        continue
                    
                    # Parse ISO 8601 to UTC datetime
                    if end_date_str.endswith('Z'):
                        end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
                    else:
                        end_date = datetime.fromisoformat(end_date_str)
                        if end_date.tzinfo is None:
                            end_date = end_date.replace(tzinfo=timezone.utc)
                    
                    # Calculate minutes until expiry
                    duration = end_date - now
                    minutes_remaining = duration.total_seconds() / 60
                    
                    # TIME FILTER: Only imminent expiries (> 0 and < max_duration)
                    if not (0 < minutes_remaining <= max_duration_minutes):
                        continue
                    
                    # Volume filter
                    volume = float(market.get('volume', 0))
                    if volume < min_volume:
                        continue
                    
                    # Parse outcomes and prices (Gamma structure)
                    outcomes_raw = market.get('outcomes', '[]')
                    prices_raw = market.get('outcomePrices', '[]')
                    
                    # Handle stringified JSON
                    if isinstance(outcomes_raw, str):
                        outcomes = json.loads(outcomes_raw)
                    else:
                        outcomes = outcomes_raw
                    
                    if isinstance(prices_raw, str):
                        prices = json.loads(prices_raw)
                    else:
                        prices = prices_raw
                    
                    # Extract YES/NO prices
                    yes_price = float(prices[0]) if len(prices) > 0 else 0.0
                    no_price = float(prices[1]) if len(prices) > 1 else 0.0
                    
                    # Calculate implied probability and edge potential
                    implied_prob = yes_price if yes_price > 0 else 0.5
                    
                    all_markets.append({
                        "expiry_min": round(minutes_remaining, 1),
                        "event": event.get('title', 'Unknown Event'),
                        "question": market.get('question', 'Unknown Question'),
                        "yes_price": yes_price,
                        "no_price": no_price,
                        "volume": volume,
                        "market_slug": market.get('slug', ''),
                        "market_id": market.get('id', ''),
                        "end_date": end_date_str,
                        "implied_prob": implied_prob,
                        "outcomes": outcomes
                    })
                    
                    batch_count += 1
                    
                except (ValueError, TypeError, KeyError) as e:
                    logger.debug(
                        "failed_to_parse_market",
                        error=str(e),
                        market_id=market.get('id'),
                        error_type=type(e).__name__
                    )
                    continue
                    
                except Exception as e:
                    logger.warning(
                        "unexpected_error_parsing_market",
                        error=str(e),
                        market_id=market.get('id')
                    )
                    continue
        
        return batch_count
    
    def get_arbitrage_candidates(
        self,
        df: pd.DataFrame,
        edge_threshold: float = 0.05
    ) -> pd.DataFrame:
        """
        Filter for markets with potential arbitrage opportunities.
        
        Critical: Polymarket charges 1-3% taker fees on 15-minute markets.
        Edge must be >4-5% to be profitable after fees.
        
        Args:
            df: DataFrame from fetch_active_15m_markets()
            edge_threshold: Minimum edge required (default: 5%)
            
        Returns:
            Filtered DataFrame with high-edge opportunities
            
        Note:
            This is a placeholder. Real edge calculation requires:
            - Real-time Binance/CoinGecko prices
            - Volatility models (GARCH, realized vol)
            - Taker fee calculations (1-3% depending on prob)
        """
        if df.empty:
            return df
        
        # Placeholder: Filter by price inefficiency
        # Real implementation needs external price source
        candidates = df[
            ((df['yes_price'] < 0.4) | (df['yes_price'] > 0.6)) &
            (df['volume'] > 100)  # Require some liquidity
        ].copy()
        
        logger.info(
            "identified_arbitrage_candidates",
            total_candidates=len(candidates),
            edge_threshold=edge_threshold
        )
        
        return candidates


def main() -> None:
    """
    CLI entry point for 15-minute crypto market scanner.
    
    This is the "Red Pill" moment - high-velocity binary options trading.
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Scan for imminent-expiry crypto binary option markets (15-minute windows)',
        epilog='⚠️  WARNING: These markets have 1-3% taker fees. Edge must be >5% to profit.'
    )
    parser.add_argument(
        '--max-duration',
        type=int,
        default=MAX_DURATION_MINUTES,
        help='Maximum minutes until expiry (default: 25)'
    )
    parser.add_argument(
        '--min-volume',
        type=float,
        default=MIN_VOLUME,
        help='Minimum volume in USD (default: 0)'
    )
    parser.add_argument(
        '--edge-threshold',
        type=float,
        default=0.05,
        help='Minimum edge for arbitrage (default: 0.05 = 5%%)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output as JSON'
    )
    parser.add_argument(
        '--arbitrage-only',
        action='store_true',
        help='Show only high-edge arbitrage candidates'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Continuous monitoring mode (refresh every 60s)'
    )
    parser.add_argument(
        '--fast',
        action='store_true',
        help='Fast scan mode (only first 10 pages, ~5 seconds)'
    )
    
    args = parser.parse_args()
    
    fetcher = Crypto15MinuteFetcher()
    
    # Watch mode: continuous monitoring
    if args.watch:
        logger.info("starting_watch_mode", refresh_interval=60)
        
        try:
            while True:
                df = fetcher.fetch_active_15m_markets(
                    max_duration_minutes=args.max_duration,
                    min_volume=args.min_volume,
                    max_pages=10 if args.fast else None
                )
                
                if not df.empty:
                    if args.arbitrage_only:
                        df = fetcher.get_arbitrage_candidates(df, args.edge_threshold)
                    
                    if not df.empty:
                        print(f"\n⚡ LIVE 15-MINUTE OPPORTUNITIES [{datetime.now().strftime('%H:%M:%S')}] ⚡")
                        print(df[['expiry_min', 'event', 'yes_price', 'volume']].to_string(index=False))
                        print(f"\nTotal: {len(df)} markets")
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] No opportunities found.")
                else:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] No active 15m markets.")
                
                sleep(60)  # Refresh every minute
                
        except KeyboardInterrupt:
            logger.info("watch_mode_stopped_by_user")
            print("\n👋 Watch mode stopped.")
            return
    
    # Single scan mode
    else:
        df = fetcher.fetch_active_15m_markets(
            max_duration_minutes=args.max_duration,
            min_volume=args.min_volume,
            max_pages=10 if args.fast else None
        )
        
        if not df.empty:
            if args.arbitrage_only:
                df = fetcher.get_arbitrage_candidates(df, args.edge_threshold)
            
            if args.json:
                print(df.to_json(orient='records', indent=2))
            else:
                print("\n💰 LIVE 15-MINUTE OPPORTUNITIES 💰\n")
                print(df[['expiry_min', 'event', 'yes_price', 'no_price', 'volume']].to_string(index=False))
                print(f"\n✅ Total: {len(df)} markets")
                print(f"📊 Avg Volume: ${df['volume'].mean():,.0f}")
                print(f"⏱️  Soonest Expiry: {df['expiry_min'].min():.1f} minutes")
                
                # Warning about fees
                print("\n⚠️  CRITICAL: Polymarket charges 1-3% taker fees on these markets.")
                print("    Your edge must be >5% to be profitable after fees.")
        else:
            print("⚠️  No 15-minute markets found in current window.")
            print(f"    (Checked for expiry < {args.max_duration} minutes)")


if __name__ == "__main__":
    main()

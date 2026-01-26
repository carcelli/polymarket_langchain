"""
Crypto event scanner for binary price prediction markets.

Scans Polymarket for markets like:
- "Will Bitcoin be above $100,000 on March 31?"
- "ETH price on December 31, 2025?"

Enriches with external price data from your crypto container.
"""

import httpx
from datetime import datetime, timezone
from typing import Optional
import logging

from ..base import EventScanner, ScanResult, Edge
from .models import (
    CryptoPriceMarket,
    PriceSignal,
    PriceDataSource,
    Asset,
    parse_strike_price,
    parse_asset,
)

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"
CRYPTO_TAG_ID = "21"


class CryptoScanner(EventScanner[CryptoPriceMarket]):
    """
    Scanner for crypto binary price prediction markets.

    Usage:
        scanner = CryptoScanner(price_source=MyPriceContainer())
        result = scanner.scan()
        enriched = scanner.enrich(result.markets)
        tradeable = scanner.filter_tradeable(enriched, min_edge=0.05)
    """

    def __init__(self, price_source: Optional[PriceDataSource] = None):
        """
        Args:
            price_source: External price data provider.
                          If None, uses built-in CoinGecko fallback.
        """
        self.price_source = price_source or CoinGeckoFallback()

    def scan(self) -> ScanResult[CryptoPriceMarket]:
        """
        Fetch all crypto price prediction markets from Polymarket.

        Returns:
            ScanResult containing CryptoPriceMarket objects.
        """
        markets = self._fetch_from_gamma()
        return ScanResult(markets=markets, source="polymarket_gamma")

    def enrich(self, markets: list[CryptoPriceMarket]) -> list[CryptoPriceMarket]:
        """
        Enrich markets with current price data from external source.

        Adds PriceSignal to each market with:
        - Current spot price
        - 24h price change
        - Volatility
        """
        enriched: list[CryptoPriceMarket] = []

        for market in markets:
            try:
                signal = self._get_price_signal(market.asset)
                market.signal = signal
                enriched.append(market)
            except Exception as e:
                logger.warning(f"Failed to enrich {market.id}: {e}")
                enriched.append(market)

        return enriched

    def filter_tradeable(
        self,
        markets: list[CryptoPriceMarket],
        min_volume: float = 1000,
        min_liquidity: float = 500,
        min_edge: float = 0.0,
    ) -> list[CryptoPriceMarket]:
        """
        Filter to markets worth trading.

        Args:
            min_volume: Minimum 24h volume in USD
            min_liquidity: Minimum liquidity in USD
            min_edge: Minimum edge magnitude (0.05 = 5%)
        """
        filtered: list[CryptoPriceMarket] = []

        for m in markets:
            if m.volume < min_volume:
                continue
            if m.liquidity < min_liquidity:
                continue

            # Calculate edge if we have price data
            if min_edge > 0 and m.signal:
                our_prob = self._estimate_probability(m)
                edge = abs(m.calculate_edge(our_prob))
                if edge < min_edge:
                    continue

            filtered.append(m)

        return sorted(filtered, key=lambda m: m.volume, reverse=True)

    def find_edge(
        self, markets: list[CryptoPriceMarket]
    ) -> list[tuple[CryptoPriceMarket, Edge]]:
        """
        Find markets with positive edge.

        Returns list of (market, edge) tuples sorted by edge magnitude.
        """
        edges = []

        for market in markets:
            if not market.signal:
                continue

            our_prob = self._estimate_probability(market)
            edge = Edge(
                market_id=market.id,
                our_prob=our_prob,
                market_prob=market.implied_prob,
            )

            if edge.edge_magnitude > 0.02:  # At least 2% edge
                edges.append((market, edge))

        return sorted(edges, key=lambda x: x[1].edge_magnitude, reverse=True)

    def _fetch_from_gamma(self) -> list[CryptoPriceMarket]:
        """Fetch crypto markets from Gamma API."""
        try:
            response = httpx.get(
                f"{GAMMA_API}/events",
                params={
                    "tag_id": CRYPTO_TAG_ID,
                    "closed": "false",
                    "limit": 200,
                },
                timeout=15,
            )
            response.raise_for_status()
            events = response.json()
        except httpx.HTTPError as e:
            logger.error(f"Gamma API error: {e}")
            return []

        markets = []
        for event in events:
            for raw in event.get("markets", []):
                market = self._parse_market(raw, event)
                if market:
                    markets.append(market)

        logger.info(f"Fetched {len(markets)} crypto markets")
        return markets

    def _parse_market(self, raw: dict, event: dict) -> Optional[CryptoPriceMarket]:
        """Parse raw API response into CryptoPriceMarket."""
        question = raw.get("question", "")

        # Must be a price prediction market
        asset = parse_asset(question)
        strike = parse_strike_price(question)

        if not asset:
            return None

        # Parse expiry
        end_date = raw.get("end_date_iso") or raw.get("endDate")
        if not end_date:
            return None

        try:
            if isinstance(end_date, str):
                if end_date.endswith("Z"):
                    expiry = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                else:
                    expiry = datetime.fromisoformat(end_date)
                    if expiry.tzinfo is None:
                        expiry = expiry.replace(tzinfo=timezone.utc)
            else:
                return None
        except (ValueError, TypeError):
            return None

        # Parse prices
        outcome_prices = raw.get("outcomePrices", [])
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = [
                    float(p.strip("'\"")) for p in outcome_prices.strip("[]").split(",")
                ]
            except (ValueError, AttributeError):
                outcome_prices = []

        yes_price = float(outcome_prices[0]) if outcome_prices else 0.5

        # Get token ID for trading
        tokens = raw.get("tokens", [])
        token_id = ""
        if tokens:
            for t in tokens:
                if t.get("outcome", "").lower() == "yes":
                    token_id = t.get("token_id", "")
                    break
            if not token_id:
                token_id = tokens[0].get("token_id", "")

        return CryptoPriceMarket(
            id=raw.get("condition_id") or raw.get("id", ""),
            question=question,
            asset=asset,
            strike_price=strike or 0,
            expiry=expiry,
            yes_price=yes_price,
            volume=float(raw.get("volume", 0)),
            liquidity=float(raw.get("liquidity", 0)),
            token_id=token_id,
            event_id=event.get("id", ""),
        )

    def _get_price_signal(self, asset: Asset) -> PriceSignal:
        """Get price signal from external data source."""
        current = self.price_source.get_current_price(asset)
        history = self.price_source.get_price_history(asset, hours=24)
        volatility = self.price_source.get_volatility(asset, hours=24)

        price_24h_ago = history[0][1] if history else current

        return PriceSignal(
            asset=asset,
            current_price=current,
            price_24h_ago=price_24h_ago,
            volatility_24h=volatility,
            timestamp=datetime.utcnow(),
        )

    def _estimate_probability(self, market: CryptoPriceMarket) -> float:
        """
        Estimate probability of price hitting strike.

        Simple model: distance to strike + trend + time.
        Replace with your ML model for better estimates.
        """
        if not market.signal or not market.strike_price:
            return 0.5

        current = market.signal.current_price
        strike = market.strike_price
        hours = market.time_to_expiry_hours
        volatility = market.signal.volatility_24h

        # Distance to strike as percentage
        distance_pct = (strike - current) / current * 100

        # Base probability from distance (adjusted by volatility)
        vol_factor = max(0.5, min(2.0, volatility / 5.0))  # Normalize around 5% vol
        if distance_pct > 0:
            # Price needs to go UP to hit strike
            # More distance = lower probability, higher vol = higher chance
            base_prob = max(0.1, 0.5 - distance_pct / (20 * vol_factor))
        else:
            # Price is already above strike
            # Needs to stay above
            base_prob = min(0.9, 0.5 + abs(distance_pct) / (20 * vol_factor))

        # Adjust for time (more time = regression to 50%)
        time_factor = min(1.0, hours / 168)  # 1 week normalization
        prob = base_prob * (1 - time_factor * 0.3) + 0.5 * time_factor * 0.3

        # Adjust for trend
        if market.signal.trend == "bullish" and distance_pct > 0:
            prob *= 1.1
        elif market.signal.trend == "bearish" and distance_pct < 0:
            prob *= 0.9

        return max(0.05, min(0.95, prob))


class CoinGeckoFallback(PriceDataSource):
    """Fallback price source using CoinGecko public API."""

    ASSET_MAP = {
        Asset.BTC: "bitcoin",
        Asset.ETH: "ethereum",
        Asset.SOL: "solana",
        Asset.XRP: "ripple",
        Asset.DOGE: "dogecoin",
    }

    def get_current_price(self, asset: Asset) -> float:
        coin_id = self.ASSET_MAP.get(asset)
        if not coin_id:
            raise ValueError(f"Unsupported asset: {asset}")

        response = httpx.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": coin_id, "vs_currencies": "usd"},
            timeout=5,
        )
        response.raise_for_status()
        return response.json()[coin_id]["usd"]

    def get_price_history(
        self, asset: Asset, hours: int = 24
    ) -> list[tuple[datetime, float]]:
        coin_id = self.ASSET_MAP.get(asset)
        if not coin_id:
            return []

        try:
            response = httpx.get(
                f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart",
                params={"vs_currency": "usd", "days": hours / 24},
                timeout=10,
            )
            response.raise_for_status()
            prices = response.json().get("prices", [])
            return [
                (datetime.fromtimestamp(ts / 1000, tz=timezone.utc), price)
                for ts, price in prices
            ]
        except Exception:
            return []

    def get_volatility(self, asset: Asset, hours: int = 24) -> float:
        history = self.get_price_history(asset, hours)
        if len(history) < 2:
            return 5.0  # Default 5% volatility

        prices = [p for _, p in history]
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        std = variance**0.5
        return (std / mean) * 100

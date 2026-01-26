"""
ML-Enhanced Crypto Scanner

Combines 15-minute market fetcher with real-time ML predictions.
Identifies opportunities where model edge exceeds fee threshold.

Usage:
    from polymarket_agents.domains.crypto.ml_scanner import CryptoMLScanner

    scanner = CryptoMLScanner()
    opportunities = scanner.scan()

    for opp in opportunities:
        print(f"{opp['asset']}: {opp['direction']} edge={opp['edge']:.1%}")
"""

import ccxt
import httpx
import numpy as np
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class Opportunity:
    """Trading opportunity with edge."""

    market_id: str
    asset: str
    question: str
    direction: str  # 'YES' or 'NO'
    model_prob: float
    market_prob: float
    edge: float
    confidence: float
    expiry_minutes: float
    volume: float
    current_price: float
    indicators: Dict


class CryptoMLScanner:
    """
    Scans 15-minute crypto markets and applies ML prediction.

    Integrates:
    - Gamma API for market discovery
    - Binance for real-time prices
    - Technical indicators for prediction
    """

    GAMMA_API = "https://gamma-api.polymarket.com"
    CRYPTO_TAG_ID = "21"
    FEE_THRESHOLD = 0.04  # 4% minimum edge to cover fees

    def __init__(self, exchange_id: str = "binance", lookback_minutes: int = 30):
        self.exchange = getattr(ccxt, exchange_id)()
        self.lookback = lookback_minutes

    def scan(
        self,
        max_duration_minutes: int = 25,
        min_volume: float = 100,
        min_edge: Optional[float] = None,
    ) -> List[Opportunity]:
        """
        Scan for profitable opportunities.

        Args:
            max_duration_minutes: Only markets expiring within this window
            min_volume: Minimum market volume in USD
            min_edge: Minimum edge (default: FEE_THRESHOLD)

        Returns:
            List of Opportunity objects sorted by edge
        """
        min_edge = min_edge or self.FEE_THRESHOLD

        # Fetch active 15-minute markets
        markets = self._fetch_markets(max_duration_minutes, min_volume)

        if not markets:
            return []

        # Analyze each market
        opportunities = []
        for market in markets:
            opp = self._analyze_market(market, min_edge)
            if opp:
                opportunities.append(opp)

        # Sort by edge descending
        opportunities.sort(key=lambda x: x.edge, reverse=True)

        return opportunities

    def _fetch_markets(
        self,
        max_duration: int,
        min_volume: float
    ) -> List[Dict]:
        """Fetch active crypto markets from Gamma API."""

        markets = []
        offset = 0
        limit = 50
        now = datetime.now(timezone.utc)

        while True:
            try:
                resp = httpx.get(
                    f"{self.GAMMA_API}/events",
                    params={
                        "limit": limit,
                        "offset": offset,
                        "closed": "false",
                        "tag_id": self.CRYPTO_TAG_ID,
                        "order": "volume24hr",
                        "ascending": "false",
                    },
                    timeout=10,
                )

                if resp.status_code != 200:
                    break

                events = resp.json()
                if not events:
                    break

                # Process markets
                for event in events:
                    for market in event.get("markets", []):
                        # Parse expiry
                        end_str = market.get("endDate")
                        if not end_str:
                            continue

                        if end_str.endswith("Z"):
                            end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                        else:
                            end = datetime.fromisoformat(end_str)
                            if end.tzinfo is None:
                                end = end.replace(tzinfo=timezone.utc)

                        minutes_left = (end - now).total_seconds() / 60

                        # Filter by time and volume
                        if not (0 < minutes_left <= max_duration):
                            continue

                        volume = float(market.get("volume", 0))
                        if volume < min_volume:
                            continue

                        # Parse prices
                        prices_raw = market.get("outcomePrices", "[]")
                        if isinstance(prices_raw, str):
                            prices = json.loads(prices_raw)
                        else:
                            prices = prices_raw

                        yes_price = float(prices[0]) if prices else 0.5
                        no_price = float(prices[1]) if len(prices) > 1 else 0.5

                        # Extract asset from question
                        question = market.get("question", "")
                        asset = self._extract_asset(question)

                        if asset:
                            markets.append({
                                "id": market.get("id"),
                                "asset": asset,
                                "question": question,
                                "yes_price": yes_price,
                                "no_price": no_price,
                                "volume": volume,
                                "expiry_min": minutes_left,
                            })

                offset += limit

                # Limit pagination
                if offset >= 500:
                    break

            except Exception as e:
                print(f"Error fetching markets: {e}")
                break

        return markets

    def _extract_asset(self, question: str) -> Optional[str]:
        """Extract asset symbol from market question."""

        q = question.lower()

        if "bitcoin" in q or "btc" in q:
            return "BTC"
        elif "ethereum" in q or "eth" in q:
            return "ETH"
        elif "solana" in q or "sol" in q:
            return "SOL"
        elif "xrp" in q:
            return "XRP"
        elif "dogecoin" in q or "doge" in q:
            return "DOGE"

        return None

    def _parse_threshold(self, question: str, asset: str) -> Optional[float]:
        """
        Parse price threshold from market question.

        Example questions:
        - "Will Bitcoin be above $87,000 at 8:00PM ET?"
        - "Bitcoin Up or Down - January 25, 7:45PM-8:00PM ET"
        """
        import re

        # Look for dollar amount pattern
        patterns = [
            r'\$([0-9,]+(?:\.[0-9]+)?)',  # $87,000 or $87,000.50
            r'above ([0-9,]+)',  # above 87000
            r'below ([0-9,]+)',  # below 87000
        ]

        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                price_str = match.group(1).replace(',', '')
                try:
                    return float(price_str)
                except ValueError:
                    continue

        return None

    def _calculate_threshold_probability(
        self,
        current_price: float,
        threshold: float,
        volatility: float,
        minutes_remaining: float,
        is_above: bool = True,
    ) -> float:
        """
        Calculate probability of price crossing threshold.

        Uses simple volatility-based model:
        - Distance to threshold in std devs
        - Time decay (shorter time = more certain)
        """
        from scipy.stats import norm

        if minutes_remaining <= 0:
            return 1.0 if (current_price > threshold) == is_above else 0.0

        # Annualized volatility to per-minute
        # Assume ~20% annual vol for BTC as baseline
        annual_vol = max(volatility * 100, 0.20)  # At least 20%
        minute_vol = annual_vol / np.sqrt(365 * 24 * 60)

        # Volatility over remaining time
        period_vol = minute_vol * np.sqrt(minutes_remaining)

        # Distance to threshold in std devs
        if threshold == 0:
            return 0.5

        pct_distance = (threshold - current_price) / current_price
        z_score = pct_distance / period_vol if period_vol > 0 else 0

        # Probability of being above threshold
        prob_above = 1 - norm.cdf(z_score)

        return prob_above if is_above else (1 - prob_above)

    def _analyze_market(
        self,
        market: Dict,
        min_edge: float
    ) -> Optional[Opportunity]:
        """Analyze market with ML prediction."""

        asset = market["asset"]
        symbol = f"{asset}/USDT"
        question = market["question"]

        try:
            # Fetch candles from exchange
            ohlcv = self.exchange.fetch_ohlcv(symbol, "1m", limit=self.lookback)

            if len(ohlcv) < self.lookback:
                return None

            # Extract price data
            closes = np.array([x[4] for x in ohlcv])
            volumes = np.array([x[5] for x in ohlcv])
            highs = np.array([x[2] for x in ohlcv])
            lows = np.array([x[3] for x in ohlcv])

            current_price = closes[-1]

            # Calculate indicators
            indicators = self._calculate_indicators(closes, volumes, highs, lows)

            # Try to parse threshold from question
            threshold = self._parse_threshold(question, asset)

            # If we have a threshold, use probability model
            if threshold:
                # Determine if this is "above" or "below" market
                is_above = "up" in question.lower() or "above" in question.lower()

                model_prob = self._calculate_threshold_probability(
                    current_price=current_price,
                    threshold=threshold,
                    volatility=indicators["volatility"],
                    minutes_remaining=market["expiry_min"],
                    is_above=is_above,
                )

                indicators["threshold"] = threshold
                indicators["distance_pct"] = (current_price - threshold) / threshold

                # YES = above threshold, NO = below
                yes_price = market["yes_price"]
                no_price = market["no_price"]

                # Calculate edge
                if is_above:
                    yes_edge = model_prob - yes_price
                    no_edge = (1 - model_prob) - no_price
                else:
                    yes_edge = (1 - model_prob) - yes_price
                    no_edge = model_prob - no_price

                if yes_edge > no_edge and yes_edge >= min_edge:
                    return Opportunity(
                        market_id=market["id"],
                        asset=asset,
                        question=question,
                        direction="YES",
                        model_prob=model_prob if is_above else (1 - model_prob),
                        market_prob=yes_price,
                        edge=yes_edge,
                        confidence=abs(model_prob - 0.5) * 2,
                        expiry_minutes=market["expiry_min"],
                        volume=market["volume"],
                        current_price=current_price,
                        indicators=indicators,
                    )
                elif no_edge > yes_edge and no_edge >= min_edge:
                    return Opportunity(
                        market_id=market["id"],
                        asset=asset,
                        question=question,
                        direction="NO",
                        model_prob=(1 - model_prob) if is_above else model_prob,
                        market_prob=no_price,
                        edge=no_edge,
                        confidence=abs(model_prob - 0.5) * 2,
                        expiry_minutes=market["expiry_min"],
                        volume=market["volume"],
                        current_price=current_price,
                        indicators=indicators,
                    )

                return None

            # Score direction
            up_prob, down_prob = self._score_direction(indicators)

            # Determine best bet
            yes_price = market["yes_price"]
            no_price = market["no_price"]

            # YES = price goes UP, NO = price goes DOWN (typically)
            # Check which side has edge
            up_edge = up_prob - yes_price
            down_edge = down_prob - no_price

            if up_edge > down_edge and up_edge >= min_edge:
                return Opportunity(
                    market_id=market["id"],
                    asset=asset,
                    question=market["question"],
                    direction="YES",
                    model_prob=up_prob,
                    market_prob=yes_price,
                    edge=up_edge,
                    confidence=abs(up_prob - down_prob),
                    expiry_minutes=market["expiry_min"],
                    volume=market["volume"],
                    current_price=current_price,
                    indicators=indicators,
                )
            elif down_edge > up_edge and down_edge >= min_edge:
                return Opportunity(
                    market_id=market["id"],
                    asset=asset,
                    question=market["question"],
                    direction="NO",
                    model_prob=down_prob,
                    market_prob=no_price,
                    edge=down_edge,
                    confidence=abs(up_prob - down_prob),
                    expiry_minutes=market["expiry_min"],
                    volume=market["volume"],
                    current_price=current_price,
                    indicators=indicators,
                )

            return None

        except Exception as e:
            print(f"Error analyzing {asset}: {e}")
            return None

    def _calculate_indicators(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> Dict:
        """Calculate technical indicators."""

        # Momentum
        momentum_5m = (closes[-5:].mean() - closes[-10:-5].mean()) / closes[-10:-5].mean()
        momentum_30m = (closes[-1] - closes[0]) / closes[0]

        # Volatility
        volatility = closes.std() / closes.mean()

        # Volume trend
        recent_vol = volumes[-5:].mean()
        avg_vol = volumes.mean()
        volume_spike = (recent_vol - avg_vol) / avg_vol if avg_vol > 0 else 0

        # RSI
        rsi = self._calculate_rsi(closes)

        # Mean deviation
        mean_price = closes.mean()
        deviation = (closes[-1] - mean_price) / mean_price

        return {
            "momentum_5m": momentum_5m,
            "momentum_30m": momentum_30m,
            "volatility": volatility,
            "volume_spike": volume_spike,
            "rsi": rsi,
            "deviation": deviation,
        }

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""

        if len(closes) < period + 1:
            return 50.0

        deltas = np.diff(closes)
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))

        avg_gain = gains[-period:].mean()
        avg_loss = losses[-period:].mean()

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _score_direction(self, indicators: Dict) -> Tuple[float, float]:
        """Score UP vs DOWN probability."""

        up = 0.5
        down = 0.5

        # Momentum (strongest signal)
        if indicators["momentum_5m"] > 0.001:
            up += 0.15
        elif indicators["momentum_5m"] < -0.001:
            down += 0.15

        # RSI reversal
        if indicators["rsi"] < 30:
            up += 0.1
        elif indicators["rsi"] > 70:
            down += 0.1

        # Mean reversion
        if indicators["volatility"] > 0.01:
            if indicators["deviation"] > 0.005:
                down += 0.1
            elif indicators["deviation"] < -0.005:
                up += 0.1

        # Volume confirmation
        if indicators["volume_spike"] > 0.2:
            if indicators["momentum_5m"] > 0:
                up += 0.05
            else:
                down += 0.05

        # Normalize
        total = up + down
        return up / total, down / total


def main():
    """CLI entry point."""

    import argparse

    parser = argparse.ArgumentParser(description="ML-enhanced crypto scanner")
    parser.add_argument("--min-edge", type=float, default=0.04, help="Minimum edge (default: 4%%)")
    parser.add_argument("--min-volume", type=float, default=100, help="Minimum volume (default: $100)")
    parser.add_argument("--max-duration", type=int, default=25, help="Max minutes to expiry")
    args = parser.parse_args()

    scanner = CryptoMLScanner()

    print("\n🔍 Scanning for ML opportunities...\n")

    opportunities = scanner.scan(
        max_duration_minutes=args.max_duration,
        min_volume=args.min_volume,
        min_edge=args.min_edge,
    )

    if not opportunities:
        print("❌ No opportunities found meeting criteria.")
        print(f"   (min_edge={args.min_edge:.0%}, min_volume=${args.min_volume:,.0f})")
        return

    print(f"✅ Found {len(opportunities)} opportunities:\n")

    for i, opp in enumerate(opportunities, 1):
        print(f"[{i}] {opp.asset} - {opp.direction}")
        print(f"    Question: {opp.question[:60]}...")
        print(f"    Edge: {opp.edge:+.1%} (model={opp.model_prob:.1%}, market={opp.market_prob:.1%})")
        print(f"    Confidence: {opp.confidence:.1%}")
        print(f"    Expiry: {opp.expiry_minutes:.1f} min | Volume: ${opp.volume:,.0f}")
        print(f"    Price: ${opp.current_price:,.2f}")
        print(f"    Indicators: RSI={opp.indicators['rsi']:.1f}, Mom={opp.indicators['momentum_5m']:+.2%}")
        print()


if __name__ == "__main__":
    main()

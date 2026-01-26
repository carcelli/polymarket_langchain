"""
Crypto agent for binary price prediction markets.

Minimal orchestration: scan -> enrich -> find edge -> recommend.
"""

from dataclasses import dataclass
from typing import Optional
import logging

from .scanner import CryptoScanner
from .models import CryptoPriceMarket, PriceDataSource
from ..base import Edge

logger = logging.getLogger(__name__)


@dataclass
class TradeRecommendation:
    """Structured trade recommendation."""

    market: CryptoPriceMarket
    edge: Edge
    action: str  # BUY_YES, BUY_NO, PASS
    size_fraction: float  # Kelly fraction
    reasoning: str


class CryptoAgent:
    """
    Agent for crypto binary price prediction markets.

    Usage:
        agent = CryptoAgent(price_source=MyPriceContainer())
        recommendations = agent.run()
        for rec in recommendations:
            if rec.action.startswith("BUY"):
                execute_trade(rec.market, rec.edge.side, rec.size_fraction)
    """

    def __init__(
        self,
        price_source: Optional[PriceDataSource] = None,
        min_volume: float = 5000,
        min_edge: float = 0.05,
        max_recommendations: int = 5,
    ):
        self.scanner = CryptoScanner(price_source=price_source)
        self.min_volume = min_volume
        self.min_edge = min_edge
        self.max_recommendations = max_recommendations

    def run(self) -> list[TradeRecommendation]:
        """
        Full pipeline: scan -> enrich -> find edge -> recommend.

        Returns top recommendations sorted by edge.
        """
        # 1. Scan
        result = self.scanner.scan()
        logger.info(f"Scanned {len(result)} crypto markets")

        if not result.markets:
            return []

        # 2. Enrich with price data
        enriched = self.scanner.enrich(result.markets)
        logger.info(f"Enriched {len(enriched)} markets with price data")

        # 3. Filter tradeable
        tradeable = self.scanner.filter_tradeable(
            enriched,
            min_volume=self.min_volume,
            min_edge=self.min_edge,
        )
        logger.info(f"Found {len(tradeable)} tradeable markets")

        # 4. Find edge
        edges = self.scanner.find_edge(tradeable)

        # 5. Build recommendations
        recommendations = []
        for market, edge in edges[: self.max_recommendations]:
            rec = self._build_recommendation(market, edge)
            recommendations.append(rec)

        return recommendations

    def scan_asset(self, asset: str) -> list[TradeRecommendation]:
        """
        Scan for specific asset (BTC, ETH, etc).

        Returns recommendations for that asset only.
        """
        from .models import Asset

        try:
            target_asset = Asset(asset.upper())
        except ValueError:
            logger.error(f"Unknown asset: {asset}")
            return []

        result = self.scanner.scan()
        enriched = self.scanner.enrich(result.markets)

        # Filter to target asset
        asset_markets = [
            m
            for m in enriched
            if isinstance(m, CryptoPriceMarket) and m.asset == target_asset
        ]

        tradeable = self.scanner.filter_tradeable(
            asset_markets,
            min_volume=self.min_volume,
            min_edge=self.min_edge,
        )

        edges = self.scanner.find_edge(tradeable)

        return [
            self._build_recommendation(m, e)
            for m, e in edges[: self.max_recommendations]
        ]

    def _build_recommendation(
        self, market: CryptoPriceMarket, edge: Edge
    ) -> TradeRecommendation:
        """Build trade recommendation from market and edge."""
        # Determine action
        if edge.edge_magnitude < self.min_edge:
            action = "PASS"
            reasoning = (
                f"Edge {edge.edge_magnitude:.1%} below threshold {self.min_edge:.1%}"
            )
        elif edge.side == "YES":
            action = "BUY_YES"
            reasoning = self._build_reasoning(market, edge, "YES")
        else:
            action = "BUY_NO"
            reasoning = self._build_reasoning(market, edge, "NO")

        return TradeRecommendation(
            market=market,
            edge=edge,
            action=action,
            size_fraction=edge.kelly_fraction() if action != "PASS" else 0,
            reasoning=reasoning,
        )

    def _build_reasoning(self, market: CryptoPriceMarket, edge: Edge, side: str) -> str:
        """Build human-readable reasoning."""
        parts = [
            f"{market.asset.value} binary prediction",
            f"Strike: ${market.strike_price:,.0f}" if market.strike_price else "",
            f"Expires in {market.time_to_expiry_hours:.1f}h",
        ]

        if market.signal:
            parts.append(f"Current price: ${market.signal.current_price:,.2f}")
            if market.distance_to_strike is not None:
                parts.append(f"Distance to strike: {market.distance_to_strike:+.1f}%")
            parts.append(f"Trend: {market.signal.trend}")

        parts.extend(
            [
                f"Our prob: {edge.our_prob:.1%}",
                f"Market prob: {edge.market_prob:.1%}",
                f"Edge: {edge.edge:+.1%}",
                f"Kelly: {edge.kelly_fraction():.1%}",
                f"Recommendation: {side}",
            ]
        )

        return " | ".join(p for p in parts if p)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Crypto prediction market agent")
    parser.add_argument("--asset", type=str, help="Filter by asset (BTC, ETH)")
    parser.add_argument("--min-volume", type=float, default=5000)
    parser.add_argument("--min-edge", type=float, default=0.05)
    parser.add_argument("--max", type=int, default=5, help="Max recommendations")

    args = parser.parse_args()

    agent = CryptoAgent(
        min_volume=args.min_volume,
        min_edge=args.min_edge,
        max_recommendations=args.max,
    )

    if args.asset:
        recommendations = agent.scan_asset(args.asset)
    else:
        recommendations = agent.run()

    if not recommendations:
        print("No recommendations found.")
        return

    print(f"\n{'='*80}")
    print(f"CRYPTO RECOMMENDATIONS ({len(recommendations)} found)")
    print(f"{'='*80}\n")

    for i, rec in enumerate(recommendations, 1):
        print(f"[{i}] {rec.market.question}")
        print(f"    Action: {rec.action}")
        print(f"    Edge: {rec.edge.edge:+.1%}")
        print(f"    Kelly: {rec.size_fraction:.1%}")
        print(f"    Reasoning: {rec.reasoning}")
        print()


if __name__ == "__main__":
    main()

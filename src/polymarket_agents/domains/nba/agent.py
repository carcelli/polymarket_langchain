"""
NBA agent for game outcomes and player props.

Minimal orchestration: scan -> enrich -> find edge -> recommend.
"""

from dataclasses import dataclass
from typing import Optional, Union
import logging

from .scanner import NBAScanner
from .models import (
    NBAGameMarket,
    NBAPlayerProp,
    SportsDataSource,
)
from ..base import Edge

logger = logging.getLogger(__name__)


@dataclass
class TradeRecommendation:
    """Structured trade recommendation."""

    market: Union[NBAGameMarket, NBAPlayerProp]
    edge: Edge
    action: str  # BUY_YES, BUY_NO, PASS
    size_fraction: float
    reasoning: str


class NBAAgent:
    """
    Agent for NBA betting markets.

    Usage:
        agent = NBAAgent(data_source=MySportsContainer())
        recommendations = agent.run()
        for rec in recommendations:
            if rec.action.startswith("BUY"):
                execute_trade(rec.market, rec.edge.side, rec.size_fraction)
    """

    def __init__(
        self,
        data_source: Optional[SportsDataSource] = None,
        min_volume: float = 5000,
        min_edge: float = 0.05,
        max_recommendations: int = 5,
    ):
        self.scanner = NBAScanner(data_source=data_source)
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
        logger.info(f"Scanned {len(result)} NBA markets")

        if not result.markets:
            return []

        # 2. Enrich with stats
        enriched = self.scanner.enrich(result.markets)
        logger.info(f"Enriched {len(enriched)} markets with stats")

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

    def scan_games(self) -> list[TradeRecommendation]:
        """Scan only game outcome markets (no props)."""
        result = self.scanner.scan()
        enriched = self.scanner.enrich(result.markets)

        # Filter to game markets only
        games = [m for m in enriched if isinstance(m, NBAGameMarket)]

        tradeable = self.scanner.filter_tradeable(
            games,
            min_volume=self.min_volume,
            min_edge=self.min_edge,
        )

        edges = self.scanner.find_edge(tradeable)

        return [
            self._build_recommendation(m, e)
            for m, e in edges[: self.max_recommendations]
        ]

    def scan_props(self) -> list[TradeRecommendation]:
        """Scan only player prop markets."""
        result = self.scanner.scan()
        enriched = self.scanner.enrich(result.markets)

        # Filter to props only
        props = [m for m in enriched if isinstance(m, NBAPlayerProp)]

        tradeable = self.scanner.filter_tradeable(
            props,
            min_volume=self.min_volume,
            min_edge=self.min_edge,
        )

        edges = self.scanner.find_edge(tradeable)

        return [
            self._build_recommendation(m, e)
            for m, e in edges[: self.max_recommendations]
        ]

    def analyze_matchup(self, home: str, away: str) -> Optional[dict]:
        """
        Analyze specific matchup without looking for market.

        Returns matchup analysis with Log5 probability.
        """
        matchup = self.scanner._build_matchup(home, away)

        return {
            "home_team": home,
            "away_team": away,
            "home_record": f"{matchup.home_team.wins}-{matchup.home_team.losses}",
            "away_record": f"{matchup.away_team.wins}-{matchup.away_team.losses}",
            "home_win_pct": matchup.home_team.win_pct,
            "away_win_pct": matchup.away_team.win_pct,
            "neutral_prob": matchup.neutral_prob,
            "home_prob": matchup.home_prob,
            "away_prob": matchup.away_prob,
            "key_factors": matchup.key_factors,
        }

    def _build_recommendation(
        self, market: Union[NBAGameMarket, NBAPlayerProp], edge: Edge
    ) -> TradeRecommendation:
        """Build trade recommendation."""
        if edge.edge_magnitude < self.min_edge:
            action = "PASS"
            reasoning = f"Edge {edge.edge_magnitude:.1%} below threshold"
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

    def _build_reasoning(
        self,
        market: Union[NBAGameMarket, NBAPlayerProp],
        edge: Edge,
        side: str,
    ) -> str:
        """Build human-readable reasoning."""
        parts = []

        if isinstance(market, NBAGameMarket):
            parts.append(f"{market.home_team} vs {market.away_team}")
            if market.matchup:
                m = market.matchup
                parts.append(
                    f"Records: {m.home_team.wins}-{m.home_team.losses} vs "
                    f"{m.away_team.wins}-{m.away_team.losses}"
                )
                parts.append(f"Log5: {m.neutral_prob:.1%} (neutral)")
                parts.append(f"Adjusted: {m.home_prob:.1%} (home)")
                if m.key_factors:
                    parts.append(f"Factors: {', '.join(m.key_factors)}")

        elif isinstance(market, NBAPlayerProp):
            parts.append(f"{market.player_name} {market.market_type.value}")
            parts.append(f"Line: {market.stat_line}")
            if market.player_stats:
                avg = getattr(
                    market.player_stats,
                    {"points": "ppg", "assists": "apg", "rebounds": "rpg"}.get(
                        market.market_type.value, "ppg"
                    ),
                    0,
                )
                parts.append(f"Avg: {avg}")

        parts.extend(
            [
                f"Our prob: {edge.our_prob:.1%}",
                f"Market: {edge.market_prob:.1%}",
                f"Edge: {edge.edge:+.1%}",
                f"Kelly: {edge.kelly_fraction():.1%}",
                f"Action: {side}",
            ]
        )

        return " | ".join(parts)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="NBA betting market agent")
    parser.add_argument(
        "--mode",
        choices=["all", "games", "props", "matchup"],
        default="all",
    )
    parser.add_argument("--home", type=str, help="Home team for matchup analysis")
    parser.add_argument("--away", type=str, help="Away team for matchup analysis")
    parser.add_argument("--min-volume", type=float, default=5000)
    parser.add_argument("--min-edge", type=float, default=0.05)
    parser.add_argument("--max", type=int, default=5)

    args = parser.parse_args()

    agent = NBAAgent(
        min_volume=args.min_volume,
        min_edge=args.min_edge,
        max_recommendations=args.max,
    )

    if args.mode == "matchup":
        if not args.home or not args.away:
            print("--home and --away required for matchup mode")
            return
        analysis = agent.analyze_matchup(args.home, args.away)
        print(f"\nMatchup: {args.home} vs {args.away}")
        for k, v in analysis.items():
            print(f"  {k}: {v}")
        return

    if args.mode == "games":
        recommendations = agent.scan_games()
    elif args.mode == "props":
        recommendations = agent.scan_props()
    else:
        recommendations = agent.run()

    if not recommendations:
        print("No recommendations found.")
        return

    print(f"\n{'='*80}")
    print(f"NBA RECOMMENDATIONS ({len(recommendations)} found)")
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

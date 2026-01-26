"""
NBA event scanner for game outcomes and player props.

Scans Polymarket for:
- Game winners (moneyline)
- Point spreads
- Player props

Enriches with team stats, injuries from your sports data container.
"""

import httpx
from datetime import datetime, timezone
from typing import Optional, Union
import logging

from ..base import EventScanner, ScanResult, Edge
from .models import (
    NBAGameMarket,
    NBAPlayerProp,
    TeamStats,
    Matchup,
    PlayerStats,
    SportsDataSource,
    MarketType,
    normalize_team,
    parse_teams,
    parse_player_prop,
)

# Type alias for NBA markets
NBAMarket = Union[NBAGameMarket, NBAPlayerProp]

logger = logging.getLogger(__name__)

GAMMA_API = "https://gamma-api.polymarket.com"

# Sports tag IDs from Polymarket
SPORTS_TAG_ID = "16"
NBA_KEYWORDS = ["nba", "basketball", "lakers", "celtics", "knicks", "warriors"]


class NBAScanner(EventScanner[NBAMarket]):
    """
    Scanner for NBA betting markets.

    Usage:
        scanner = NBAScanner(data_source=MySportsContainer())
        result = scanner.scan()
        enriched = scanner.enrich(result.markets)
        edges = scanner.find_edge(enriched)
    """

    HOME_COURT_ADVANTAGE = 0.06  # ~6% historical home court edge

    def __init__(self, data_source: Optional[SportsDataSource] = None):
        """
        Args:
            data_source: External sports data provider.
                        If None, uses built-in static standings.
        """
        self.data_source = data_source or StaticStandings()

    def scan(self) -> ScanResult[NBAMarket]:
        """
        Fetch all NBA markets from Polymarket.

        Returns:
            ScanResult containing NBAGameMarket and NBAPlayerProp objects.
        """
        markets = self._fetch_from_gamma()
        return ScanResult(markets=markets, source="polymarket_gamma")

    def enrich(self, markets: list[NBAMarket]) -> list[NBAMarket]:
        """
        Enrich markets with team/player stats from external source.

        For games: Adds Matchup with Log5 probability.
        For props: Adds PlayerStats with averages.
        """
        enriched: list[NBAMarket] = []

        for market in markets:
            try:
                if isinstance(market, NBAGameMarket):
                    market.matchup = self._build_matchup(
                        market.home_team, market.away_team
                    )
                    enriched.append(market)
                elif isinstance(market, NBAPlayerProp):
                    market.player_stats = self.data_source.get_player_stats(
                        market.player_name
                    )
                    enriched.append(market)
            except Exception as e:
                logger.warning(f"Failed to enrich {market.id}: {e}")
                enriched.append(market)

        return enriched

    def filter_tradeable(
        self,
        markets: list[NBAMarket],
        min_volume: float = 1000,
        min_liquidity: float = 500,
        min_edge: float = 0.0,
    ) -> list[NBAMarket]:
        """
        Filter to markets worth trading.

        Args:
            min_volume: Minimum 24h volume
            min_liquidity: Minimum liquidity
            min_edge: Minimum edge magnitude (0.05 = 5%)
        """
        filtered: list[NBAMarket] = []

        for m in markets:
            if m.volume < min_volume:
                continue
            if m.liquidity < min_liquidity:
                continue

            if min_edge > 0:
                if isinstance(m, NBAGameMarket) and m.matchup:
                    edge = abs(m.calculate_edge(m.matchup.home_prob))
                    if edge < min_edge:
                        continue
                elif isinstance(m, NBAPlayerProp) and m.player_stats:
                    our_prob = self._estimate_prop_probability(m)
                    edge = abs(m.calculate_edge(our_prob))
                    if edge < min_edge:
                        continue

            filtered.append(m)

        return sorted(filtered, key=lambda m: m.volume, reverse=True)

    def find_edge(
        self, markets: list[Union[NBAGameMarket, NBAPlayerProp]]
    ) -> list[tuple[Union[NBAGameMarket, NBAPlayerProp], Edge]]:
        """
        Find markets with positive edge.

        Returns list of (market, edge) sorted by edge magnitude.
        """
        edges = []

        for market in markets:
            if isinstance(market, NBAGameMarket) and market.matchup:
                our_prob = market.matchup.home_prob
                edge = Edge(
                    market_id=market.id,
                    our_prob=our_prob,
                    market_prob=market.implied_prob,
                )
            elif isinstance(market, NBAPlayerProp) and market.player_stats:
                our_prob = self._estimate_prop_probability(market)
                edge = Edge(
                    market_id=market.id,
                    our_prob=our_prob,
                    market_prob=market.implied_prob,
                )
            else:
                continue

            if edge.edge_magnitude > 0.03:  # At least 3% edge
                edges.append((market, edge))

        return sorted(edges, key=lambda x: x[1].edge_magnitude, reverse=True)

    def _fetch_from_gamma(self) -> list[Union[NBAGameMarket, NBAPlayerProp]]:
        """Fetch NBA markets from Gamma API."""
        try:
            response = httpx.get(
                f"{GAMMA_API}/events",
                params={
                    "tag_id": SPORTS_TAG_ID,
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
            # Filter to NBA only
            title = event.get("title", "").lower()
            if not any(kw in title for kw in NBA_KEYWORDS):
                continue

            for raw in event.get("markets", []):
                market = self._parse_market(raw, event)
                if market:
                    markets.append(market)

        logger.info(f"Fetched {len(markets)} NBA markets")
        return markets

    def _parse_market(
        self, raw: dict, event: dict
    ) -> Optional[Union[NBAGameMarket, NBAPlayerProp]]:
        """Parse raw API response into market object."""
        question = raw.get("question", "")

        # Try to parse as game market
        home, away = parse_teams(question)
        if home and away:
            return self._parse_game_market(raw, event, home, away)

        # Try to parse as player prop
        player, line, prop_type = parse_player_prop(question)
        if player and line and prop_type is not None:
            return self._parse_player_prop(raw, event, player, line, prop_type)

        return None

    def _parse_game_market(
        self, raw: dict, event: dict, home: str, away: str
    ) -> Optional[NBAGameMarket]:
        """Parse game outcome market."""
        # Parse game date
        end_date = raw.get("end_date_iso") or raw.get("endDate")
        if not end_date:
            return None

        try:
            if isinstance(end_date, str):
                if end_date.endswith("Z"):
                    game_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                else:
                    game_date = datetime.fromisoformat(end_date)
                    if game_date.tzinfo is None:
                        game_date = game_date.replace(tzinfo=timezone.utc)
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

        # Get token ID
        tokens = raw.get("tokens", [])
        token_id = tokens[0].get("token_id", "") if tokens else ""

        return NBAGameMarket(
            id=raw.get("condition_id") or raw.get("id", ""),
            question=raw.get("question", ""),
            market_type=MarketType.MONEYLINE,
            home_team=home,
            away_team=away,
            game_date=game_date,
            yes_price=yes_price,
            volume=float(raw.get("volume", 0)),
            liquidity=float(raw.get("liquidity", 0)),
            token_id=token_id,
            event_id=event.get("id", ""),
        )

    def _parse_player_prop(
        self,
        raw: dict,
        event: dict,
        player: str,
        line: float,
        prop_type: MarketType,
    ) -> Optional[NBAPlayerProp]:
        """Parse player prop market."""
        end_date = raw.get("end_date_iso") or raw.get("endDate")
        if not end_date:
            return None

        try:
            if isinstance(end_date, str):
                if end_date.endswith("Z"):
                    game_date = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
                else:
                    game_date = datetime.fromisoformat(end_date)
                    if game_date.tzinfo is None:
                        game_date = game_date.replace(tzinfo=timezone.utc)
            else:
                return None
        except (ValueError, TypeError):
            return None

        outcome_prices = raw.get("outcomePrices", [])
        if isinstance(outcome_prices, str):
            try:
                outcome_prices = [
                    float(p.strip("'\"")) for p in outcome_prices.strip("[]").split(",")
                ]
            except (ValueError, AttributeError):
                outcome_prices = []

        yes_price = float(outcome_prices[0]) if outcome_prices else 0.5

        tokens = raw.get("tokens", [])
        token_id = tokens[0].get("token_id", "") if tokens else ""

        return NBAPlayerProp(
            id=raw.get("condition_id") or raw.get("id", ""),
            question=raw.get("question", ""),
            market_type=prop_type,
            player_name=player,
            team="",  # Would need external lookup
            opponent="",
            game_date=game_date,
            stat_line=line,
            yes_price=yes_price,
            volume=float(raw.get("volume", 0)),
            liquidity=float(raw.get("liquidity", 0)),
            token_id=token_id,
            event_id=event.get("id", ""),
        )

    def _build_matchup(self, home: str, away: str) -> Matchup:
        """Build matchup analysis using Log5 formula."""
        home_stats = self.data_source.get_team_stats(home)
        away_stats = self.data_source.get_team_stats(away)

        # Log5 formula for neutral court probability
        neutral_prob = self._log5(home_stats.win_pct, away_stats.win_pct)

        # Adjust for home court
        home_prob = min(0.95, neutral_prob + self.HOME_COURT_ADVANTAGE)

        # Collect key factors
        factors = []
        if home_stats.streak >= 3:
            factors.append(f"{home} on {home_stats.streak}-game win streak")
        if away_stats.streak <= -3:
            factors.append(f"{away} on {abs(away_stats.streak)}-game losing streak")
        if home_stats.back_to_back:
            factors.append(f"{home} on back-to-back")
            home_prob -= 0.03
        if away_stats.back_to_back:
            factors.append(f"{away} on back-to-back")
            home_prob += 0.03

        return Matchup(
            home_team=home_stats,
            away_team=away_stats,
            neutral_prob=neutral_prob,
            home_prob=home_prob,
            key_factors=factors,
        )

    def _log5(self, team_a_pct: float, team_b_pct: float) -> float:
        """
        Bill James Log5 formula.

        Probability of A beating B on neutral court.
        """
        # Clamp to avoid division issues
        a = max(0.001, min(0.999, team_a_pct))
        b = max(0.001, min(0.999, team_b_pct))

        numerator = a * (1 - b)
        denominator = numerator + b * (1 - a)

        return numerator / denominator if denominator > 0 else 0.5

    def _estimate_prop_probability(self, prop: NBAPlayerProp) -> float:
        """
        Estimate probability of player hitting the line.

        Simple model: compare line to player's average.
        Replace with your ML model for better estimates.
        """
        if not prop.player_stats:
            return 0.5

        stats = prop.player_stats

        if prop.market_type == MarketType.PLAYER_POINTS:
            avg = stats.ppg
        elif prop.market_type == MarketType.PLAYER_ASSISTS:
            avg = stats.apg
        elif prop.market_type == MarketType.PLAYER_REBOUNDS:
            avg = stats.rpg
        else:
            return 0.5

        if avg == 0:
            return 0.5

        # Distance from line as percentage of average
        distance = (prop.stat_line - avg) / avg

        # Base probability: 50% if line equals average
        # Adjust based on distance (roughly 15% swing per 20% distance)
        prob = 0.5 - distance * 0.75

        return max(0.1, min(0.9, prob))


class StaticStandings(SportsDataSource):
    """
    Fallback data source with static standings.

    Replace with your sports data container for live data.
    """

    # 2024-25 season standings (update regularly)
    STANDINGS = {
        # Eastern Conference
        "Cavaliers": (36, 6),
        "Celtics": (31, 13),
        "Knicks": (28, 16),
        "Magic": (25, 21),
        "Bucks": (23, 19),
        "Heat": (21, 20),
        "Pacers": (24, 20),
        "76ers": (15, 25),
        "Hawks": (21, 21),
        "Bulls": (18, 24),
        "Pistons": (20, 22),
        "Nets": (14, 28),
        "Raptors": (9, 34),
        "Hornets": (8, 31),
        "Wizards": (6, 33),
        # Western Conference
        "Thunder": (35, 7),
        "Grizzlies": (30, 15),
        "Rockets": (28, 14),
        "Mavericks": (25, 21),
        "Warriors": (21, 21),
        "Lakers": (23, 18),
        "Nuggets": (25, 17),
        "Clippers": (23, 19),
        "Suns": (20, 21),
        "Timberwolves": (22, 20),
        "Kings": (19, 22),
        "Spurs": (19, 21),
        "Trail Blazers": (14, 28),
        "Jazz": (10, 31),
        "Pelicans": (12, 30),
    }

    def get_team_stats(self, team: str) -> TeamStats:
        normalized = normalize_team(team)
        if normalized in self.STANDINGS:
            wins, losses = self.STANDINGS[normalized]
            return TeamStats(name=normalized, wins=wins, losses=losses)
        return TeamStats(name=team, wins=0, losses=0)

    def get_player_stats(self, player: str) -> PlayerStats:
        # Would need external API for real player stats
        return PlayerStats(name=player, team="")

    def get_injuries(self, team: str) -> list[str]:
        return []

    def is_back_to_back(self, team: str, game_date: datetime) -> bool:
        return False

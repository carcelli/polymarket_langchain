"""
NBA domain models.

Game outcomes: "Will the Lakers beat the Celtics?"
Player props: "Will LeBron score 25+ points?"
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Protocol
import re


class MarketType(str, Enum):
    MONEYLINE = "moneyline"  # Team to win
    SPREAD = "spread"  # Point spread
    PLAYER_POINTS = "points"  # Player total points
    PLAYER_ASSISTS = "assists"  # Player total assists
    PLAYER_REBOUNDS = "rebounds"  # Player total rebounds
    PLAYER_COMBO = "combo"  # Points + Rebounds + Assists


@dataclass
class TeamStats:
    """Team statistics for edge calculation."""

    name: str
    wins: int
    losses: int
    home_wins: int = 0
    home_losses: int = 0
    away_wins: int = 0
    away_losses: int = 0
    streak: int = 0  # Positive = winning streak, negative = losing
    back_to_back: bool = False  # Playing second game in 2 days

    @property
    def win_pct(self) -> float:
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.5

    @property
    def home_win_pct(self) -> float:
        total = self.home_wins + self.home_losses
        return self.home_wins / total if total > 0 else 0.55  # Default home advantage

    @property
    def away_win_pct(self) -> float:
        total = self.away_wins + self.away_losses
        return self.away_wins / total if total > 0 else 0.45


@dataclass
class Matchup:
    """Pre-computed matchup analysis."""

    home_team: TeamStats
    away_team: TeamStats
    neutral_prob: float  # Log5 probability (neutral court)
    home_prob: float  # Adjusted for home court
    key_factors: list[str] = field(default_factory=list)

    @property
    def away_prob(self) -> float:
        return 1 - self.home_prob


@dataclass
class NBAGameMarket:
    """
    NBA game outcome market.

    Example: "Will the Lakers beat the Celtics on Jan 20?"
    """

    id: str
    question: str
    market_type: MarketType
    home_team: str
    away_team: str
    game_date: datetime
    yes_price: float
    volume: float
    liquidity: float
    token_id: str
    event_id: str

    # Optional spread for spread markets
    spread: Optional[float] = None

    # Enriched from external data
    matchup: Optional[Matchup] = None

    @property
    def implied_prob(self) -> float:
        """Market's implied probability."""
        return self.yes_price

    @property
    def hours_until_game(self) -> float:
        delta = self.game_date - datetime.utcnow()
        return max(0, delta.total_seconds() / 3600)

    def calculate_edge(self, our_prob: float) -> float:
        """Edge = our probability - market probability."""
        return our_prob - self.implied_prob


@dataclass
class PlayerStats:
    """Player statistics for prop betting."""

    name: str
    team: str
    ppg: float = 0.0  # Points per game
    apg: float = 0.0  # Assists per game
    rpg: float = 0.0  # Rebounds per game
    minutes: float = 0.0  # Minutes per game
    games_played: int = 0
    injured: bool = False
    injury_note: str = ""


@dataclass
class NBAPlayerProp:
    """
    NBA player prop market.

    Example: "Will LeBron score 25+ points vs Celtics?"
    """

    id: str
    question: str
    market_type: MarketType
    player_name: str
    team: str
    opponent: str
    game_date: datetime
    stat_line: float  # The over/under line (e.g., 25.5 points)
    yes_price: float
    volume: float
    liquidity: float
    token_id: str
    event_id: str

    # Enriched from external data
    player_stats: Optional[PlayerStats] = None

    @property
    def implied_prob(self) -> float:
        return self.yes_price

    def calculate_edge(self, our_prob: float) -> float:
        return our_prob - self.implied_prob


class SportsDataSource(Protocol):
    """
    Protocol for external NBA data.

    Implement this to connect to your sports data container.
    """

    def get_team_stats(self, team: str) -> TeamStats:
        """Get current team statistics."""
        ...

    def get_player_stats(self, player: str) -> PlayerStats:
        """Get player statistics."""
        ...

    def get_injuries(self, team: str) -> list[str]:
        """Get list of injured players for team."""
        ...

    def is_back_to_back(self, team: str, game_date: datetime) -> bool:
        """Check if team is playing back-to-back."""
        ...


# Team name normalization (Polymarket uses various formats)
TEAM_ALIASES = {
    # City names
    "los angeles lakers": "Lakers",
    "la lakers": "Lakers",
    "boston celtics": "Celtics",
    "new york knicks": "Knicks",
    "golden state warriors": "Warriors",
    "miami heat": "Heat",
    "phoenix suns": "Suns",
    "milwaukee bucks": "Bucks",
    "denver nuggets": "Nuggets",
    "dallas mavericks": "Mavericks",
    "philadelphia 76ers": "76ers",
    "philly": "76ers",
    "cleveland cavaliers": "Cavaliers",
    "cavs": "Cavaliers",
    "memphis grizzlies": "Grizzlies",
    "oklahoma city thunder": "Thunder",
    "okc": "Thunder",
    "houston rockets": "Rockets",
    "sacramento kings": "Kings",
    "indiana pacers": "Pacers",
    "orlando magic": "Magic",
    "atlanta hawks": "Hawks",
    "chicago bulls": "Bulls",
    "brooklyn nets": "Nets",
    "toronto raptors": "Raptors",
    "charlotte hornets": "Hornets",
    "washington wizards": "Wizards",
    "detroit pistons": "Pistons",
    "minnesota timberwolves": "Timberwolves",
    "wolves": "Timberwolves",
    "utah jazz": "Jazz",
    "portland trail blazers": "Trail Blazers",
    "blazers": "Trail Blazers",
    "new orleans pelicans": "Pelicans",
    "san antonio spurs": "Spurs",
    "la clippers": "Clippers",
    "los angeles clippers": "Clippers",
}


def normalize_team(name: str) -> str:
    """Normalize team name to standard format."""
    lower = name.lower().strip()

    # Check aliases
    if lower in TEAM_ALIASES:
        return TEAM_ALIASES[lower]

    # Check if it's already a standard name
    standard_names = set(TEAM_ALIASES.values())
    for std in standard_names:
        if std.lower() == lower:
            return std

    # Return capitalized if no match
    return name.title()


def parse_teams(question: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract teams from market question.

    Examples:
        "Will the Lakers beat the Celtics?" -> ("Lakers", "Celtics")
        "Knicks vs Mavericks - who wins?" -> ("Knicks", "Mavericks")
    """
    q = question.lower()

    # Pattern: "Team A vs Team B" or "Team A beat Team B"
    patterns = [
        r"will (?:the )?(\w+) beat (?:the )?(\w+)",
        r"(\w+) vs\.? (\w+)",
        r"(\w+) at (\w+)",  # Away at Home
    ]

    for pattern in patterns:
        match = re.search(pattern, q)
        if match:
            team1 = normalize_team(match.group(1))
            team2 = normalize_team(match.group(2))
            return team1, team2

    return None, None


def parse_player_prop(
    question: str,
) -> tuple[Optional[str], Optional[float], Optional[MarketType]]:
    """
    Extract player name and line from prop question.

    Examples:
        "Will LeBron James score 25+ points?" -> ("LeBron James", 25.0, PLAYER_POINTS)
        "Steph Curry over 6.5 assists?" -> ("Steph Curry", 6.5, PLAYER_ASSISTS)
    """
    q = question.lower()

    # Points pattern
    points_match = re.search(
        r"(\w+(?:\s+\w+)?)\s+(?:score|over|under)\s+(\d+\.?\d*)\+?\s*points?", q
    )
    if points_match:
        return (
            points_match.group(1).title(),
            float(points_match.group(2)),
            MarketType.PLAYER_POINTS,
        )

    # Assists pattern
    assists_match = re.search(
        r"(\w+(?:\s+\w+)?)\s+(?:over|under)\s+(\d+\.?\d*)\s*assists?", q
    )
    if assists_match:
        return (
            assists_match.group(1).title(),
            float(assists_match.group(2)),
            MarketType.PLAYER_ASSISTS,
        )

    # Rebounds pattern
    rebounds_match = re.search(
        r"(\w+(?:\s+\w+)?)\s+(?:over|under)\s+(\d+\.?\d*)\s*rebounds?", q
    )
    if rebounds_match:
        return (
            rebounds_match.group(1).title(),
            float(rebounds_match.group(2)),
            MarketType.PLAYER_REBOUNDS,
        )

    return None, None, None

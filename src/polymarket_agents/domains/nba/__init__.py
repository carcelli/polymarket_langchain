"""
NBA domain for game outcomes and player props.

Focuses on:
- Game winners (moneyline)
- Point spreads
- Player props (points, assists, rebounds)

External data: Team stats, injuries, schedules from sports data container.

Usage:
    from polymarket_agents.domains.nba import NBAAgent
    agent = NBAAgent(data_source=my_container)
    recommendations = agent.run()
"""

from .agent import NBAAgent, TradeRecommendation
from .models import Matchup, NBAGameMarket, NBAPlayerProp, SportsDataSource, TeamStats
from .scanner import NBAScanner

__all__ = [
    "Matchup",
    "NBAAgent",
    "NBAGameMarket",
    "NBAPlayerProp",
    "NBAScanner",
    "SportsDataSource",
    "TeamStats",
    "TradeRecommendation",
]

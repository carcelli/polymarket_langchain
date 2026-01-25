#!/usr/bin/env python3
"""
NBA Game Winner Predictor

Baseline: Log5 formula + home advantage (~6% historical edge).
Extensible to injuries, rest days, Elo ratings.

This is your edge-proving ground - sports has better signal than crypto.
"""

from typing import Tuple, Optional, Dict
import math

# Current NBA standings (2024-25 season as of mid-January)
# Format: {"Team": (wins, losses)}
NBA_STANDINGS = {
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


class NBAPredictor:
    """
    Baseline NBA win probability predictor.

    Uses:
    - Team winning percentage
    - Log5 formula (Bill James)
    - Home court advantage (~6% boost)
    """

    def __init__(self, standings: Dict[str, Tuple[int, int]] = None):
        """
        Args:
            standings: Dict of team -> (wins, losses). Uses default if None.
        """
        self.standings = standings or NBA_STANDINGS

    def get_win_percentage(self, team: str) -> float:
        """Get team's current win percentage."""
        if team not in self.standings:
            # Unknown team - return league average
            return 0.500

        wins, losses = self.standings[team]
        games = wins + losses

        if games == 0:
            return 0.500

        return wins / games

    def log5_probability(self, team_a_pct: float, team_b_pct: float) -> float:
        """
        Calculate win probability using Log5 formula.

        Log5 = (A * (1-B)) / (A * (1-B) + B * (1-A))

        Where A and B are win percentages.
        """
        if team_a_pct >= 0.999:
            team_a_pct = 0.999
        if team_a_pct <= 0.001:
            team_a_pct = 0.001
        if team_b_pct >= 0.999:
            team_b_pct = 0.999
        if team_b_pct <= 0.001:
            team_b_pct = 0.001

        numerator = team_a_pct * (1 - team_b_pct)
        denominator = numerator + team_b_pct * (1 - team_a_pct)

        return numerator / denominator if denominator > 0 else 0.5

    def predict_winner(
        self, team_a: str, team_b: str, is_team_a_home: bool = True
    ) -> Tuple[str, float, Dict]:
        """
        Predict game winner with probability.

        Args:
            team_a: First team name
            team_b: Second team name
            is_team_a_home: Whether team_a is home team

        Returns:
            (predicted_winner, probability, details_dict)
        """

        # Get win percentages
        pct_a = self.get_win_percentage(team_a)
        pct_b = self.get_win_percentage(team_b)

        # Calculate Log5 probability (neutral court)
        neutral_prob_a = self.log5_probability(pct_a, pct_b)

        # Apply home court advantage (~6% boost historically)
        HOME_ADVANTAGE = 0.060

        if is_team_a_home:
            adjusted_prob_a = neutral_prob_a + HOME_ADVANTAGE
        else:
            adjusted_prob_a = neutral_prob_a - HOME_ADVANTAGE

        # Clamp to [1%, 99%] for safety
        adjusted_prob_a = max(0.01, min(0.99, adjusted_prob_a))

        # Determine winner
        if adjusted_prob_a > 0.5:
            winner = team_a
            confidence = adjusted_prob_a
        else:
            winner = team_b
            confidence = 1 - adjusted_prob_a

        # Build details
        details = {
            "team_a": team_a,
            "team_b": team_b,
            "team_a_record": self.standings.get(team_a, (0, 0)),
            "team_b_record": self.standings.get(team_b, (0, 0)),
            "team_a_pct": pct_a,
            "team_b_pct": pct_b,
            "neutral_prob_a": neutral_prob_a,
            "home_advantage": HOME_ADVANTAGE if is_team_a_home else -HOME_ADVANTAGE,
            "adjusted_prob_a": adjusted_prob_a,
            "predicted_winner": winner,
            "win_probability": confidence,
        }

        return winner, confidence, details

    def calculate_edge(
        self, team: str, opponent: str, is_home: bool, market_price: float
    ) -> Tuple[float, str]:
        """
        Calculate betting edge vs market price.

        Args:
            team: Team to bet on
            opponent: Opposing team
            is_home: Whether team is home
            market_price: Current market price (0-1, e.g., 0.75 = 75¢)

        Returns:
            (edge, recommendation)
            edge > 0 means team is undervalued
            recommendation in ['BUY', 'PASS', 'AVOID']
        """

        winner, model_prob, details = self.predict_winner(team, opponent, is_home)

        # Our edge = model probability - market price
        if winner == team:
            edge = model_prob - market_price
        else:
            edge = (1 - model_prob) - market_price

        # Recommendation thresholds
        if edge > 0.05:  # 5%+ edge
            recommendation = "BUY"
        elif edge < -0.05:  # Significantly overpriced
            recommendation = "AVOID"
        else:
            recommendation = "PASS"

        return edge, recommendation


def demo_predictions():
    """Demo predictions for current games."""

    predictor = NBAPredictor()

    print("🏀 NBA Predictor Demo - Baseline Log5 + Home Advantage\n")
    print("=" * 80)

    # Example games (your list from Polymarket)
    games = [
        ("Knicks", "Mavericks", True, 0.79),  # Knicks home, market 79¢
        ("76ers", "Pacers", True, 0.75),  # 76ers home, market 75¢
        ("Suns", "Nets", False, 0.75),  # Suns away, Nets market 26¢ = Suns 74¢
        ("Pistons", "Celtics", True, 0.60),  # Pistons home, market 60¢
        ("Warriors", "Heat", True, 0.69),  # Warriors home, market 69¢
    ]

    for home_team, away_team, _, market_price in games:
        winner, prob, details = predictor.predict_winner(
            home_team, away_team, is_team_a_home=True
        )

        edge, rec = predictor.calculate_edge(home_team, away_team, True, market_price)

        print(f"\n{home_team} vs {away_team} (Home: {home_team})")
        print(
            f"  Records: {home_team} {details['team_a_record']} ({details['team_a_pct']:.3f}) | "
            f"{away_team} {details['team_b_record']} ({details['team_b_pct']:.3f})"
        )
        print(f"  Model: {winner} wins with {prob:.1%} probability")
        print(f"  Market: {home_team} at {market_price:.1%}")
        print(f"  Edge: {edge:+.1%} → {rec}")

        if rec == "BUY":
            print(f"  💡 SIGNAL: Bet {home_team} - undervalued by {abs(edge):.1%}")
        elif rec == "AVOID":
            print(f"  ⚠️  WARNING: {home_team} overpriced by {abs(edge):.1%}")
        else:
            print(f"  ⏭️  PASS: Edge too small")


if __name__ == "__main__":
    demo_predictions()

    print("\n\n💡 Next Steps:")
    print("   1. Fetch live markets: python scripts/nba_market_fetcher.py")
    print("   2. Run simulator: python scripts/nba_simulator.py")
    print("   3. Enhance predictor:")
    print("      - Add injury data (web scraping)")
    print("      - Add rest days (back-to-back games)")
    print("      - Add Elo ratings (538-style)")
    print("      - Train ML model on historical data")

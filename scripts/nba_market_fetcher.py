#!/usr/bin/env python3
"""
NBA Market Fetcher for Polymarket

Fetches active NBA game winner markets with volume, prices, implied probabilities.
Superior to crypto Up/Down: longer horizons, abundant features, documented inefficiencies.
"""

import requests
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple

API_BASE = "https://gamma-api.polymarket.com"


def fetch_active_sports_markets(category_filter: str = None) -> List[Dict]:
    """
    Fetch all active markets, filter for sports/NBA.

    Args:
        category_filter: Optional category like "Sports", "NBA"

    Returns:
        List of market dicts with key fields
    """
    url = f"{API_BASE}/markets"
    params = {
        "limit": 500,
        "closed": False,
        "active": True,
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        # API returns array directly or wrapped in 'data'
        all_markets = response.json()
        if isinstance(all_markets, dict) and "data" in all_markets:
            all_markets = all_markets["data"]

        nba_markets = []
        for market in all_markets:
            question = market.get("question", "")
            category = market.get("category", "")

            # Filter for NBA/basketball markets
            if not (
                "NBA" in category
                or "NBA" in question
                or "basketball" in question.lower()
                or any(
                    team in question
                    for team in [
                        "Lakers",
                        "Celtics",
                        "Warriors",
                        "Heat",
                        "Knicks",
                        "76ers",
                        "Suns",
                        "Mavericks",
                        "Nets",
                        "Pistons",
                        "Pacers",
                        "Bucks",
                        "Nuggets",
                        "Clippers",
                    ]
                )
            ):
                continue

            # Extract outcome prices
            outcomes = market.get("outcomes", [])
            outcome_prices = market.get("outcome_prices", [])

            # Parse prices
            yes_price = None
            no_price = None

            if outcome_prices:
                try:
                    # outcome_prices is usually string like "['0.79', '0.21']"
                    if isinstance(outcome_prices, str):
                        prices = json.loads(outcome_prices.replace("'", '"'))
                    else:
                        prices = outcome_prices

                    if len(prices) >= 1:
                        yes_price = float(prices[0])
                    if len(prices) >= 2:
                        no_price = float(prices[1])
                except:
                    pass

            # Extract tokens
            tokens = []
            clob_token_ids = market.get("clob_token_ids", "")
            if clob_token_ids:
                try:
                    if isinstance(clob_token_ids, str):
                        tokens = json.loads(clob_token_ids.replace("'", '"'))
                    else:
                        tokens = clob_token_ids
                except:
                    pass

            nba_markets.append(
                {
                    "id": market.get("condition_id") or market.get("id"),
                    "question": question,
                    "category": category,
                    "volume": float(market.get("volume", 0)),
                    "liquidity": float(market.get("liquidity", 0)),
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "end_date": market.get("end_date_iso") or market.get("end_date"),
                    "tokens": tokens,
                    "outcomes": outcomes,
                }
            )

        return sorted(nba_markets, key=lambda x: x["volume"], reverse=True)

    except Exception as e:
        print(f"Error fetching markets: {e}")
        return []


def extract_game_info(market: Dict) -> Dict:
    """
    Parse question for teams, date, matchup info.

    Handles common formats:
    - "Will the Knicks beat the Mavericks on January 19?"
    - "Knicks vs Mavericks - January 19"
    - "Winner: Knicks vs Mavericks"
    """
    question = market["question"]

    # Common NBA teams (abbreviations and full names)
    teams = {
        "Lakers": "LAL",
        "Celtics": "BOS",
        "Warriors": "GSW",
        "Heat": "MIA",
        "Knicks": "NYK",
        "76ers": "PHI",
        "Sixers": "PHI",
        "Suns": "PHX",
        "Mavericks": "DAL",
        "Mavs": "DAL",
        "Nets": "BKN",
        "Pistons": "DET",
        "Pacers": "IND",
        "Bucks": "MIL",
        "Nuggets": "DEN",
        "Clippers": "LAC",
        "Bulls": "CHI",
        "Raptors": "TOR",
        "Hawks": "ATL",
        "Hornets": "CHA",
        "Jazz": "UTA",
        "Thunder": "OKC",
        "Trail Blazers": "POR",
        "Blazers": "POR",
        "Kings": "SAC",
        "Spurs": "SAS",
        "Timberwolves": "MIN",
        "Wolves": "MIN",
        "Pelicans": "NOP",
        "Magic": "ORL",
        "Wizards": "WAS",
        "Grizzlies": "MEM",
        "Rockets": "HOU",
        "Cavaliers": "CLE",
        "Cavs": "CLE",
    }

    found_teams = []
    for team, abbr in teams.items():
        if team in question:
            found_teams.append((team, abbr))

    # Determine favorite (team with Yes in question or higher price)
    favorite = None
    underdog = None

    if len(found_teams) >= 2:
        # First mentioned team is usually the favorite in "Will X beat Y" format
        if "beat" in question.lower() or "defeat" in question.lower():
            favorite = found_teams[0][0]
            underdog = found_teams[1][0]
        else:
            # Use prices to determine
            if market["yes_price"] and market["yes_price"] > 0.5:
                favorite = found_teams[0][0]
                underdog = found_teams[1][0]
            else:
                favorite = found_teams[1][0]
                underdog = found_teams[0][0]
    elif len(found_teams) == 1:
        # Only one team mentioned - probably the favorite
        favorite = found_teams[0][0]

    # Extract date
    date_match = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}",
        question,
    )
    game_date = date_match.group(0) if date_match else "Unknown"

    return {
        "raw_question": question,
        "teams_found": [t[0] for t in found_teams],
        "favorite": favorite,
        "underdog": underdog,
        "date": game_date,
        "matchup": (
            f"{found_teams[0][0]} vs {found_teams[1][0]}"
            if len(found_teams) >= 2
            else "Unknown"
        ),
    }


def format_market_display(market: Dict) -> str:
    """Format market for display."""
    info = extract_game_info(market)

    volume_str = (
        f"${market['volume']/1000:.1f}k"
        if market["volume"] > 1000
        else f"${market['volume']:.0f}"
    )

    yes_price_str = f"{market['yes_price']*100:.0f}¢" if market["yes_price"] else "N/A"
    no_price_str = f"{market['no_price']*100:.0f}¢" if market["no_price"] else "N/A"

    output = []
    output.append(f"Question: {market['question']}")
    output.append(f"Matchup: {info['matchup']}")
    output.append(f"Date: {info['date']}")
    output.append(f"Volume: {volume_str}")
    output.append(f"Prices: Yes {yes_price_str} / No {no_price_str}")
    output.append(
        f"Implied Prob: {market['yes_price']:.1%}" if market["yes_price"] else ""
    )
    output.append(f"Market ID: {market['id']}")

    return "\n".join(output)


if __name__ == "__main__":
    print("🏀 Fetching active NBA markets from Polymarket...\n")

    markets = fetch_active_sports_markets()

    if not markets:
        print("❌ No NBA markets found.")
        print("   Try again during NBA season (October-June)")
        exit(0)

    print(f"✅ Found {len(markets)} active NBA markets\n")
    print("=" * 80)
    print("TOP MARKETS BY VOLUME")
    print("=" * 80)

    for i, market in enumerate(markets[:10], 1):
        print(f"\n{i}. {format_market_display(market)}")
        print("-" * 80)

    # Summary stats
    total_volume = sum(m["volume"] for m in markets)
    print("\n📊 Summary:")
    print(f"   Total markets: {len(markets)}")
    print(f"   Total volume: ${total_volume/1000:.1f}k")
    print(f"   Avg volume: ${total_volume/len(markets)/1000:.1f}k" if markets else "")
    print(
        f"   Highest volume: ${max(m['volume'] for m in markets)/1000:.1f}k"
        if markets
        else ""
    )

    print("\n💡 Next steps:")
    print("   1. Run NBA simulator: python scripts/nba_simulator.py")
    print("   2. Add predictions with team records and home advantage")
    print("   3. Track virtual P&L over 20+ games")

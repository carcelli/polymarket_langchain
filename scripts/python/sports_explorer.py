#!/usr/bin/env python3
"""
Polymarket Sports Data Explorer

Explore sports leagues, teams, market types, and related markets on Polymarket.

Usage:
    python scripts/python/sports_explorer.py --leagues           # List all leagues
    python scripts/python/sports_explorer.py --teams             # List all teams
    python scripts/python/sports_explorer.py --teams --league nfl    # Teams by league
    python scripts/python/sports_explorer.py --markets --league nba  # Markets by league
    python scripts/python/sports_explorer.py --market-types      # List market types
"""

import argparse
import json
import time
from typing import Dict, List, Optional
import httpx

GAMMA_URL = "https://gamma-api.polymarket.com"


def fetch_market_types() -> List[str]:
    """Fetch all valid sports market types."""
    response = httpx.get(f"{GAMMA_URL}/sports/market-types", timeout=30)
    response.raise_for_status()
    data = response.json()
    return data.get('marketTypes', [])


def fetch_sports() -> List[Dict]:
    """Fetch all sports/leagues metadata."""
    response = httpx.get(f"{GAMMA_URL}/sports", timeout=30)
    response.raise_for_status()
    sports = response.json()
    # Filter valid entries
    return [s for s in sports if s.get('sport')]


def fetch_teams(
    limit: int = 500,
    league: Optional[str] = None,
    offset: int = 0
) -> List[Dict]:
    """Fetch teams with optional league filter."""
    params = {"limit": limit, "offset": offset}
    if league:
        params["league"] = league
    
    response = httpx.get(f"{GAMMA_URL}/teams", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def fetch_markets_by_tag(tag_id: int, limit: int = 20) -> List[Dict]:
    """Fetch active markets for a given tag."""
    params = {
        "tag_id": tag_id,
        "closed": "false",
        "limit": limit,
        "order": "volume",
        "ascending": "false"
    }
    response = httpx.get(f"{GAMMA_URL}/events", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def display_leagues():
    """Display all available sports/leagues."""
    print("\n" + "=" * 80)
    print("  🏆 POLYMARKET SPORTS & LEAGUES")
    print("=" * 80)
    
    sports = fetch_sports()
    print(f"\n📊 Found {len(sports)} leagues\n")
    
    # Categorize by type
    traditional = []
    esports = []
    cricket = []
    soccer = []
    other = []
    
    esports_codes = {'cs2', 'csgo', 'lol', 'dota2', 'val', 'valorant', 'mlbb', 'codmw', 'rl', 'ow', 'r6siege', 'fifa', 'hok', 'pubg', 'sc', 'sc2', 'wildrift'}
    cricket_codes = {'ipl', 'odi', 't20', 'test', 'crint', 'crind', 'craus', 'crnew', 'crpak', 'creng', 'crsou', 'cruae', 'csa', 'bnd', 'ecc', 'cpl', 'bpl', 'abb', 'lpl', 'psp', 'sasa', 'she'}
    soccer_codes = {'epl', 'lal', 'bun', 'sea', 'fl1', 'ucl', 'uel', 'mls', 'ere', 'por', 'cdr', 'dfb', 'itc', 'efa', 'efl', 'cde', 'lib', 'sud', 'cof', 'lcs', 'mex', 'col', 'tur', 'spl', 'rus', 'nor', 'den', 'jap', 'kor', 'chi', 'ind', 'bra', 'arg', 'aus', 'ja2', 'acn', 'afc', 'caf', 'uef', 'fif', 'ofc', 'scop'}
    traditional_codes = {'nfl', 'nba', 'mlb', 'nhl', 'mma', 'ncaab', 'cfb', 'wnba', 'atp', 'wta', 'kbo', 'cbb', 'cwbb'}
    
    for s in sports:
        code = s.get('sport', '').lower()
        if code in esports_codes:
            esports.append(s)
        elif code in cricket_codes:
            cricket.append(s)
        elif code in soccer_codes:
            soccer.append(s)
        elif code in traditional_codes:
            traditional.append(s)
        else:
            other.append(s)
    
    def print_category(title: str, items: List[Dict]):
        if not items:
            return
        print(f"\n{'─' * 80}")
        print(f"  {title}")
        print(f"{'─' * 80}")
        print(f"{'Code':<10} {'Tags':<30} {'Resolution Source':<35}")
        print("-" * 80)
        for s in sorted(items, key=lambda x: x.get('sport', '')):
            code = s.get('sport', '')[:10]
            tags = s.get('tags', '')[:30]
            res = (s.get('resolution', '') or '')
            # Shorten URL
            res = res.replace('https://', '').replace('www.', '')[:35]
            print(f"{code:<10} {tags:<30} {res}")
    
    print_category("🏈 TRADITIONAL SPORTS", traditional)
    print_category("⚽ SOCCER/FOOTBALL", soccer)
    print_category("🏏 CRICKET", cricket)
    print_category("🎮 ESPORTS", esports)
    print_category("📌 OTHER", other)
    
    print("\n" + "=" * 80)
    print("  Use --teams --league <code> to see teams for a specific league")
    print("  Use --markets --league <code> to see active markets")
    print("=" * 80 + "\n")


def display_teams(league: Optional[str] = None, limit: int = 100):
    """Display teams, optionally filtered by league."""
    print("\n" + "=" * 80)
    if league:
        print(f"  📋 TEAMS IN {league.upper()}")
    else:
        print("  📋 ALL POLYMARKET TEAMS")
    print("=" * 80)
    
    teams = fetch_teams(limit=limit, league=league)
    
    if not teams:
        print(f"\n⚠️  No teams found" + (f" for league '{league}'" if league else ""))
        return
    
    print(f"\n📊 Found {len(teams)} teams\n")
    
    # Group by league if showing all
    if not league:
        from collections import Counter
        leagues = Counter(t.get('league', 'Unknown') for t in teams)
        
        # Show summary
        print("Teams by League:")
        for lg, count in sorted(leagues.items(), key=lambda x: -x[1])[:20]:
            print(f"  {lg:<10} {count:>4} teams")
        
        if len(leagues) > 20:
            print(f"  ... and {len(leagues) - 20} more leagues")
        
        print("\nUse --league <code> to see specific league teams")
    else:
        # Show detailed team list
        print(f"{'Abbr':<8} {'Name':<40} {'Record':<10}")
        print("-" * 60)
        for t in sorted(teams, key=lambda x: x.get('name', '') or ''):
            abbr = t.get('abbreviation', '')[:8]
            name = (t.get('name', '') or 'N/A')[:40]
            record = t.get('record', '') or ''
            print(f"{abbr:<8} {name:<40} {record:<10}")


def display_market_types():
    """Display all available sports market types."""
    print("\n" + "=" * 80)
    print("  📊 SPORTS MARKET TYPES")
    print("=" * 80)
    
    market_types = fetch_market_types()
    print(f"\n📊 Found {len(market_types)} market types\n")
    
    # Categorize market types
    categories = {
        "General": ["moneyline", "spreads", "totals", "parlays", "double_chance", "correct_score", "child_moneyline"],
        "Team Totals": ["team_totals", "team_totals_home", "team_totals_away"],
        "First Half": ["first_half_moneyline", "first_half_spreads", "first_half_totals"],
        "Football": ["anytime_touchdowns", "first_touchdowns", "two_plus_touchdowns", "passing_yards", "passing_touchdowns", "receiving_yards", "receptions", "rushing_yards"],
        "Basketball": ["points", "rebounds", "assists", "assists_points_rebounds", "threes", "double_doubles"],
        "Soccer": ["total_goals", "both_teams_to_score", "match_handicap"],
        "Baseball": ["nrfi"],
        "Tennis": ["total_games", "tennis_first_set_totals", "tennis_match_totals", "tennis_set_handicap", "tennis_first_set_winner", "tennis_set_totals"],
        "Esports (MOBA)": ["moba_first_blood", "moba_first_tower", "moba_first_dragon", "moba_total_kills"],
        "Esports (FPS)": ["shooter_rounds_total", "shooter_round_handicap", "shooter_first_pistol_round", "shooter_second_pistol_round", "map_handicap", "map_participant_win_total", "map_participant_win_one"],
        "UFC/MMA": ["ufc_go_the_distance", "ufc_method_of_victory"],
        "Cricket": ["cricket_toss_winner", "cricket_completed_match", "cricket_toss_match_double", "cricket_most_sixes", "cricket_team_top_batter", "cricket_match_to_go_till"],
    }
    
    categorized = set()
    for cat, types in categories.items():
        matching = [t for t in types if t in market_types]
        if matching:
            print(f"\n{'─' * 60}")
            print(f"  {cat}")
            print(f"{'─' * 60}")
            for t in matching:
                print(f"  • {t}")
                categorized.add(t)
    
    # Show uncategorized
    uncategorized = [t for t in market_types if t not in categorized]
    if uncategorized:
        print(f"\n{'─' * 60}")
        print(f"  Other")
        print(f"{'─' * 60}")
        for t in uncategorized:
            print(f"  • {t}")
    
    print("\n" + "=" * 80)
    print("  Use: ?sportsMarketTypes=moneyline,spreads to filter markets")
    print("=" * 80 + "\n")


def display_markets_for_league(league: str, limit: int = 10):
    """Display active markets for a league."""
    print("\n" + "=" * 80)
    print(f"  📈 ACTIVE MARKETS - {league.upper()}")
    print("=" * 80)
    
    # Find tag for this league
    sports = fetch_sports()
    sport_data = next((s for s in sports if s.get('sport', '').lower() == league.lower()), None)
    
    if not sport_data:
        print(f"\n⚠️  League '{league}' not found. Use --leagues to see available leagues.")
        return
    
    tags_str = sport_data.get('tags', '')
    if not tags_str:
        print(f"\n⚠️  No tags found for league '{league}'")
        return
    
    # Use first non-1 tag (1 is often a generic sports tag)
    tags = [int(t.strip()) for t in tags_str.split(',') if t.strip()]
    tag_id = tags[1] if len(tags) > 1 else tags[0]
    
    print(f"\n🔖 Using tag_id={tag_id}")
    print(f"📰 Resolution source: {sport_data.get('resolution', 'N/A')}")
    
    events = fetch_markets_by_tag(tag_id, limit=limit)
    
    if not events:
        print(f"\n⚠️  No active markets found for {league.upper()}")
        return
    
    print(f"\n📊 Found {len(events)} events with active markets\n")
    
    for i, event in enumerate(events, 1):
        title = event.get('title', 'N/A')[:60]
        volume = float(event.get('volume', 0) or 0)
        end_date = event.get('endDate', '')[:10]
        
        markets = event.get('markets', [])
        market_count = len(markets)
        
        print(f"\n{i}. {title}")
        print(f"   Volume: ${volume:,.0f} | Markets: {market_count} | End: {end_date}")
        
        # Show top markets in this event
        for m in markets[:3]:
            q = m.get('question', '')[:50]
            prices = m.get('outcomePrices', '[]')
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except:
                    prices = []
            
            if prices and len(prices) >= 2:
                try:
                    yes = float(prices[0]) * 100
                    no = float(prices[1]) * 100
                    print(f"   • {q}... YES: {yes:.0f}% | NO: {no:.0f}%")
                except:
                    print(f"   • {q}...")


def main():
    parser = argparse.ArgumentParser(
        description="Explore Polymarket sports data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sports_explorer.py --leagues                  # List all leagues
  python sports_explorer.py --teams                    # List all teams  
  python sports_explorer.py --teams --league nfl       # NFL teams only
  python sports_explorer.py --markets --league nba     # NBA markets
        """
    )
    
    parser.add_argument('--leagues', action='store_true', help='List all sports/leagues')
    parser.add_argument('--teams', action='store_true', help='List teams')
    parser.add_argument('--markets', action='store_true', help='Show active markets')
    parser.add_argument('--market-types', action='store_true', help='List all sports market types')
    parser.add_argument('--league', type=str, help='Filter by league code (e.g., nfl, nba)')
    parser.add_argument('--limit', type=int, default=100, help='Max results to show')
    
    args = parser.parse_args()
    
    if not any([args.leagues, args.teams, args.markets, args.market_types]):
        # Default: show leagues
        display_leagues()
        return
    
    if args.leagues:
        display_leagues()
    
    if args.market_types:
        display_market_types()
    
    if args.teams:
        display_teams(league=args.league, limit=args.limit)
    
    if args.markets:
        if not args.league:
            print("⚠️  --markets requires --league <code>")
            print("   Example: --markets --league nba")
            return
        display_markets_for_league(args.league, limit=args.limit)


if __name__ == "__main__":
    main()


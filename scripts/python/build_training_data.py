"""
Build Real ML Training Dataset from Polymarket CLOB API

The CLOB API at clob.polymarket.com/markets?closed=true exposes:
  - tokens[].winner: true/false — ground truth resolution outcome
  - tokens[].price: 0 or 1 after resolution
  - Full market metadata (volume, liquidity, end_date, question, etc.)

This is the correct supervised learning signal. The Gamma API
(gamma-api.polymarket.com/markets?closed=true) lacks the winner field
for old FPMM markets, so we use CLOB instead.

Usage:
    python scripts/python/build_training_data.py
    python scripts/python/build_training_data.py --pages 20 --output data/ml_training_dataset.parquet
"""
import sys
import os
import argparse
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import httpx
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

CLOB_URL = "https://clob.polymarket.com"
HEADERS = {"User-Agent": "PolymarketML/1.0", "Accept": "application/json"}


def fetch_clob_resolved_markets(max_pages: int = 10, page_size: int = 100) -> list[dict]:
    """
    Fetch resolved binary markets from the CLOB API with pagination.

    Returns markets where `tokens[0].winner` is defined — these are the only
    records usable as supervised-learning training labels.
    """
    markets: list[dict] = []
    next_cursor = ""
    page = 0

    logger.info(f"Fetching CLOB resolved markets (up to {max_pages} pages × {page_size})...")

    with httpx.Client(timeout=30.0, headers=HEADERS) as client:
        while page < max_pages:
            params: dict = {"closed": "true", "limit": page_size}
            if next_cursor:
                params["next_cursor"] = next_cursor

            try:
                resp = client.get(f"{CLOB_URL}/markets", params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.error(f"CLOB API error on page {page}: {exc}")
                break

            batch = data.get("data", []) if isinstance(data, dict) else data
            if not batch:
                break

            for m in batch:
                tokens = m.get("tokens", [])
                # Only include binary markets with resolved outcomes
                if len(tokens) == 2 and any(t.get("winner") is not None for t in tokens):
                    markets.append(m)

            logger.info(
                f"  Page {page + 1}: +{len(batch)} markets fetched, "
                f"{len(markets)} with resolution data"
            )

            next_cursor = data.get("next_cursor", "") if isinstance(data, dict) else ""
            if not next_cursor or len(batch) < page_size:
                break

            page += 1
            time.sleep(0.2)  # rate limit

    logger.info(f"Total resolved binary markets: {len(markets)}")
    return markets


def market_to_row(m: dict) -> dict | None:
    """
    Convert a CLOB market dict to a flat feature row with a ground-truth label.

    Target: will_resolve_yes = 1 if the YES/first token won, else 0.
    """
    tokens = m.get("tokens", [])
    if len(tokens) < 2:
        return None

    # Find yes/no tokens (YES is typically outcome="Yes" or first token)
    yes_token = next((t for t in tokens if t.get("outcome", "").lower() in ("yes", "true")), tokens[0])
    no_token  = next((t for t in tokens if t is not yes_token), tokens[1])

    # Ground truth
    winner = yes_token.get("winner")
    if winner is None:
        return None
    will_resolve_yes = int(bool(winner))

    # Pre-resolution price of YES token (0-1 during trading, becomes 0 or 1 after resolution)
    # We can't recover the pre-resolution price from the CLOB closed endpoint directly,
    # so we use the yes_token.price only if it's between 0.01 and 0.99 (active market).
    raw_price = float(yes_token.get("price", 0.5))
    # After resolution price will be 0 or 1; use 0.5 as default for the model input
    yes_price = raw_price if 0.01 < raw_price < 0.99 else 0.5

    question  = str(m.get("question", ""))
    category  = str(m.get("category", "other")).lower()
    volume    = float(m.get("volume_num", m.get("volumeNum", m.get("volume", 0))) or 0)
    liquidity = float(m.get("liquidityNum", m.get("liquidity", 0)) or 0)

    # Time-based features
    try:
        import dateutil.parser
        end_dt  = dateutil.parser.parse(m.get("end_date_iso") or m.get("endDate") or "")
        creat_dt = dateutil.parser.parse(m.get("created_at") or m.get("createdAt") or "")
        days_to_resolve = max((end_dt - creat_dt).days, 0)
    except Exception:
        days_to_resolve = 90

    return {
        # Metadata (excluded from model features)
        "market_id": m.get("condition_id", m.get("id", "")),
        "question":  question,
        "category":  category,
        "active":    m.get("active", False),
        "resolved":  True,
        "actual_outcome": will_resolve_yes,
        "will_resolve_yes": will_resolve_yes,
        # Price features
        "yes_price":                 yes_price,
        "no_price":                  1 - yes_price,
        "implied_probability":       yes_price,
        "price_distance_from_fair":  abs(yes_price - 0.5),
        "price_volatility":          yes_price * (1 - yes_price),
        "price_extremity":           int(yes_price < 0.2 or yes_price > 0.8),
        "spread":                    abs(yes_price - (1 - yes_price)),
        "market_efficiency_score":   1.0 / (abs(yes_price - (1 - yes_price)) + 0.01),
        # Volume features
        "volume":             volume,
        "liquidity":          liquidity,
        "volume_to_liquidity": volume / max(liquidity, 1),
        "log_volume":         np.log(volume + 1),
        # Time features
        "market_age_days":   days_to_resolve,
        "days_until_end":    0.0,
        "days_to_resolve":   float(days_to_resolve),
        # Text features
        "question_length":    float(len(question)),
        "description_length": 0.0,
        "has_description":    0,
        "question_word_count": float(len(question.split())),
        # Category flags
        "is_politics":    int(any(w in question.lower() for w in ["election", "president", "trump", "biden", "senate", "congress", "vote"])),
        "is_sports":      int(any(w in category for w in ["sport", "nba", "nfl", "nhl", "ncaab", "ncaaf", "mlb"]) or any(w in question.lower() for w in ["nba", "nfl", "super bowl", "mlb"])),
        "is_crypto":      int("crypto" in category or any(w in question.lower() for w in ["bitcoin", "ethereum", "btc", "eth", "crypto"])),
        "is_geopolitics": int(any(w in question.lower() for w in ["war", "nato", "ukraine", "russia", "china", "geopolit"])),
        "is_tech":        int("tech" in category),
        "is_economics":   int(any(w in question.lower() for w in ["fed", "rate", "gdp", "inflation", "recession"])),
        # Keyword flags
        "has_political_keywords": int(any(w in question.lower() for w in ["trump", "biden", "election", "president"])),
        "has_sports_keywords":    int(any(w in question.lower() for w in ["nba", "nfl", "super bowl", "mlb", "nhl", "ncaab"])),
        "has_crypto_keywords":    int(any(w in question.lower() for w in ["bitcoin", "ethereum", "crypto", "btc", "eth"])),
        # Momentum placeholders (not available from closed-market snapshot)
        "price_momentum_24h": 0.0,
        "volume_trend_7d":    0.0,
    }


def build_dataset(max_pages: int = 10, min_volume: float = 100.0) -> pd.DataFrame:
    markets = fetch_clob_resolved_markets(max_pages=max_pages)
    rows = []
    for m in markets:
        row = market_to_row(m)
        if row and row["volume"] >= min_volume:
            rows.append(row)

    if not rows:
        logger.error("No usable rows — check API connectivity")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    yes_rate = df["will_resolve_yes"].mean()
    logger.info(f"Dataset: {len(df)} rows, YES rate = {yes_rate:.1%}, "
                f"volume median = ${df['volume'].median():,.0f}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Build real ML training data from CLOB API")
    parser.add_argument("--pages",   type=int,   default=10,    help="API pages to fetch (100 markets each)")
    parser.add_argument("--min-vol", type=float, default=100.0, help="Min volume threshold")
    parser.add_argument("--output",  type=str,   default="data/ml_training_dataset.parquet")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent.parent)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df = build_dataset(max_pages=args.pages, min_volume=args.min_vol)
    if df.empty:
        logger.error("No training data built. Aborting.")
        sys.exit(1)

    df.to_parquet(args.output, index=False)
    logger.info(f"Saved → {args.output}  ({len(df)} rows)")

    # Category breakdown
    if "is_sports" in df.columns:
        print("\nCategory breakdown:")
        for flag in ["is_sports", "is_politics", "is_crypto", "is_tech"]:
            n = df[flag].sum()
            if n > 0:
                yr = df[df[flag] == 1]["will_resolve_yes"].mean()
                print(f"  {flag:<22} {n:>5} markets  YES rate {yr:.1%}")


if __name__ == "__main__":
    main()

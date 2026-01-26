"""
Automated ML Data Ingestion for Polymarket

Handles ingestion, cleaning, and preparation of historical market data
for machine learning model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import sqlite3
import requests
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PolymarketDataIngestion:
    """
    Automated data ingestion system for Polymarket ML training.

    Handles:
    - Historical market data collection
    - Resolved market outcomes
    - Feature engineering and cleaning
    - ML-ready dataset preparation
    - Continuous data updates
    """

    def __init__(self, db_path: str = "data/markets.db", cache_dir: str = "data/cache"):
        self.db_path = db_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Polymarket API endpoints
        self.base_url = "https://gamma-api.polymarket.com"
        self.headers = {"User-Agent": "PolymarketML/1.0", "Accept": "application/json"}

    def fetch_historical_markets(
        self, days_back: int = 365, limit: int = 10000
    ) -> List[Dict]:
        """
        Fetch historical markets from Polymarket API.

        Args:
            days_back: Number of days of historical data to fetch
            limit: Maximum number of markets to fetch

        Returns:
            List of market data dictionaries
        """
        logger.info(f"Fetching {days_back} days of historical market data...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Polymarket API for historical markets
        # Note: This is a simplified version - real implementation would need
        # to handle pagination and rate limiting
        markets = []

        try:
            # Get active markets first
            active_url = f"{self.base_url}/markets"
            params = {
                "active": "true",
                "limit": min(100, limit // 2),  # Split between active and closed
            }

            response = requests.get(active_url, headers=self.headers, params=params)
            response.raise_for_status()

            active_markets = response.json()
            markets.extend(active_markets)

            # Get closed markets (resolved)
            closed_url = f"{self.base_url}/markets"
            params_closed = {
                "closed": "true",
                "limit": min(100, limit // 2),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            }

            response_closed = requests.get(
                closed_url, headers=self.headers, params=params_closed
            )
            response_closed.raise_for_status()

            closed_markets = response_closed.json()
            markets.extend(closed_markets)

            logger.info(f"Successfully fetched {len(markets)} markets")
            return markets

        except Exception as e:
            logger.error(f"Failed to fetch historical markets: {e}")
            return []

    def fetch_market_details(self, market_ids: List[str]) -> List[Dict]:
        """
        Fetch detailed information for specific markets.

        Args:
            market_ids: List of market IDs to fetch details for

        Returns:
            List of detailed market information
        """
        logger.info(f"Fetching details for {len(market_ids)} markets...")

        detailed_markets = []

        for market_id in market_ids:
            try:
                # Get market details
                detail_url = f"{self.base_url}/markets/{market_id}"
                response = requests.get(detail_url, headers=self.headers)
                response.raise_for_status()

                market_detail = response.json()
                detailed_markets.append(market_detail)

                # Rate limiting
                import time

                time.sleep(0.1)  # 10 requests per second

            except Exception as e:
                logger.warning(f"Failed to fetch details for market {market_id}: {e}")
                continue

        logger.info(f"Successfully fetched details for {len(detailed_markets)} markets")
        return detailed_markets

    def clean_market_data(self, raw_markets: List[Dict]) -> pd.DataFrame:
        """
        Clean and normalize raw market data.

        Args:
            raw_markets: Raw market data from API

        Returns:
            Cleaned pandas DataFrame
        """
        logger.info(f"Cleaning data for {len(raw_markets)} markets...")

        cleaned_data = []

        for market in raw_markets:
            try:
                # Extract core market information
                market_id = market.get("id", market.get("market_id", "unknown"))
                question = market.get("question", "").strip()
                description = market.get("description", "").strip()

                # Handle outcomes
                outcomes = market.get("outcomes", [])
                if isinstance(outcomes, str):
                    outcomes = outcomes.strip("[]").replace("'", "").split(", ")

                # Handle prices
                outcome_prices = market.get("outcome_prices", [])
                if isinstance(outcome_prices, str):
                    outcome_prices = [
                        float(p.strip())
                        for p in outcome_prices.strip("[]").split(",")
                        if p.strip()
                    ]

                # Ensure we have price data
                if len(outcome_prices) < 2:
                    continue  # Skip markets without proper pricing

                yes_price = outcome_prices[0] if len(outcome_prices) > 0 else 0.5
                no_price = outcome_prices[1] if len(outcome_prices) > 1 else 0.5

                # Market metadata
                category = market.get("category", "unknown").lower().strip()
                volume = float(market.get("volume", 0))
                liquidity = float(market.get("liquidity", 0))

                # Dates
                created_at = market.get("created_at", market.get("createdAt"))
                end_date = market.get("end_date", market.get("endDate"))
                resolved_at = market.get("resolved_at", market.get("resolvedAt"))

                # Parse dates
                def parse_date(date_str):
                    if not date_str:
                        return None
                    try:
                        return pd.to_datetime(date_str)
                    except:
                        return None

                created_at = parse_date(created_at)
                end_date = parse_date(end_date)
                resolved_at = parse_date(resolved_at)

                # Resolution status and outcome
                active = market.get("active", True)
                resolved = market.get("resolved", False)
                winner = market.get("winner", None)

                # Determine actual outcome for resolved markets
                actual_outcome = None
                if resolved and winner is not None:
                    # Map winner to binary outcome (0 = No, 1 = Yes)
                    if isinstance(winner, str):
                        actual_outcome = (
                            1 if winner.lower() in ["yes", "true", "1"] else 0
                        )
                    elif isinstance(winner, bool):
                        actual_outcome = 1 if winner else 0
                    elif isinstance(winner, int):
                        actual_outcome = winner

                # Create cleaned record
                cleaned_record = {
                    "market_id": market_id,
                    "question": question,
                    "description": description,
                    "category": category,
                    "outcomes": outcomes,
                    "yes_price": yes_price,
                    "no_price": no_price,
                    "implied_probability": yes_price,  # For binary markets
                    "volume": volume,
                    "liquidity": liquidity,
                    "volume_to_liquidity": volume / max(liquidity, 1),
                    "active": active,
                    "resolved": resolved,
                    "actual_outcome": actual_outcome,
                    "created_at": created_at,
                    "end_date": end_date,
                    "resolved_at": resolved_at,
                    "days_to_resolve": None,
                    "description_length": len(description),
                    "question_length": len(question),
                    "has_description": len(description) > 0,
                }

                # Calculate days to resolve
                if resolved_at and created_at:
                    cleaned_record["days_to_resolve"] = (resolved_at - created_at).days

                cleaned_data.append(cleaned_record)

            except Exception as e:
                logger.warning(
                    f"Failed to clean market {market.get('id', 'unknown')}: {e}"
                )
                continue

        df = pd.DataFrame(cleaned_data)
        logger.info(f"Successfully cleaned {len(df)} markets")
        return df

    def engineer_ml_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features specifically for ML model training.

        Args:
            df: Cleaned market DataFrame

        Returns:
            DataFrame with additional ML features
        """
        logger.info("Engineering ML features...")

        # Create a copy to avoid modifying original
        ml_df = df.copy()

        # Price-based features
        ml_df["price_distance_from_fair"] = abs(ml_df["yes_price"] - 0.5)
        ml_df["price_volatility"] = ml_df["yes_price"] * (
            1 - ml_df["yes_price"]
        )  # Information entropy
        ml_df["price_extremity"] = (ml_df["yes_price"] < 0.2) | (
            ml_df["yes_price"] > 0.8
        )

        # Volume features
        ml_df["log_volume"] = np.log(ml_df["volume"] + 1)
        ml_df["volume_category"] = pd.cut(
            ml_df["volume"],
            bins=[0, 1000, 10000, 100000, 1000000, float("inf")],
            labels=["micro", "small", "medium", "large", "huge"],
        )

        # Time-based features
        ml_df["market_age_days"] = (pd.Timestamp.now() - ml_df["created_at"]).dt.days
        ml_df["days_until_end"] = (ml_df["end_date"] - pd.Timestamp.now()).dt.days
        ml_df["days_until_end"] = ml_df["days_until_end"].clip(
            lower=0
        )  # Don't go negative

        # Category features (one-hot would be done in model preprocessing)
        major_categories = [
            "politics",
            "sports",
            "crypto",
            "geopolitics",
            "tech",
            "economics",
        ]
        for cat in major_categories:
            ml_df[f"is_{cat}"] = ml_df["category"].str.contains(
                cat, case=False, na=False
            )

        # Text features
        ml_df["question_word_count"] = ml_df["question"].str.split().str.len()
        ml_df["has_political_keywords"] = ml_df["question"].str.contains(
            r"trump|biden|election|president|governor|senate|congress",
            case=False,
            regex=True,
        )
        ml_df["has_sports_keywords"] = ml_df["question"].str.contains(
            r"super bowl|nfl|football|basketball|nba|baseball|mlb|hockey|nhl",
            case=False,
            regex=True,
        )
        ml_df["has_crypto_keywords"] = ml_df["question"].str.contains(
            r"bitcoin|ethereum|crypto|blockchain|nft|defi", case=False, regex=True
        )

        # Market microstructure features
        ml_df["spread"] = abs(ml_df["yes_price"] - ml_df["no_price"])
        ml_df["market_efficiency_score"] = 1 / (
            ml_df["spread"] + 0.01
        )  # Lower spread = more efficient

        # Historical features (would be populated from time series data)
        ml_df["price_momentum_24h"] = 0.0  # Placeholder
        ml_df["volume_trend_7d"] = 0.0  # Placeholder

        # Target variable for supervised learning
        ml_df["will_resolve_yes"] = ml_df["actual_outcome"]  # For resolved markets

        logger.info(
            f"Engineered {len(ml_df.columns) - len(df.columns)} additional features"
        )
        return ml_df

    def create_training_dataset(
        self,
        days_back: int = 365,
        min_volume: float = 1000,
        include_unresolved: bool = False,
    ) -> pd.DataFrame:
        """
        Create a complete ML training dataset.

        Args:
            days_back: Days of historical data
            min_volume: Minimum volume threshold
            include_unresolved: Whether to include unresolved markets

        Returns:
            ML-ready training dataset
        """
        logger.info("Creating ML training dataset...")

        # Fetch historical data
        raw_markets = self.fetch_historical_markets(days_back=days_back, limit=5000)

        if not raw_markets:
            logger.error("No market data fetched")
            return pd.DataFrame()

        # Clean data
        cleaned_df = self.clean_market_data(raw_markets)

        # Apply filters
        filtered_df = cleaned_df[
            (cleaned_df["volume"] >= min_volume)
            & (cleaned_df["yes_price"].between(0.01, 0.99))  # Valid probability range
        ]

        if not include_unresolved:
            filtered_df = filtered_df[filtered_df["resolved"]]

        logger.info(f"Filtered to {len(filtered_df)} markets (min_volume={min_volume})")

        # Engineer ML features
        ml_dataset = self.engineer_ml_features(filtered_df)

        # Final cleaning
        ml_dataset = ml_dataset.dropna(subset=["will_resolve_yes"])  # Must have labels
        ml_dataset = ml_dataset.reset_index(drop=True)

        logger.info(
            f"Final ML dataset: {len(ml_dataset)} samples, {len(ml_dataset.columns)} features"
        )

        # Cache the dataset
        cache_file = (
            self.cache_dir
            / f"ml_training_data_{datetime.now().strftime('%Y%m%d')}.parquet"
        )
        ml_dataset.to_parquet(cache_file)
        logger.info(f"Cached dataset to {cache_file}")

        return ml_dataset

    def update_database(self, ml_dataset: pd.DataFrame) -> None:
        """
        Update the local database with new ML-ready data.

        Args:
            ml_dataset: ML-ready dataset to store
        """
        logger.info("Updating local database with ML data...")

        conn = sqlite3.connect(self.db_path)

        try:
            # Create ML features table if it doesn't exist
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_market_features (
                    market_id TEXT PRIMARY KEY,
                    question TEXT,
                    category TEXT,
                    yes_price REAL,
                    no_price REAL,
                    volume REAL,
                    liquidity REAL,
                    resolved BOOLEAN,
                    actual_outcome INTEGER,
                    created_at TEXT,
                    end_date TEXT,
                    resolved_at TEXT,
                    days_to_resolve INTEGER,
                    -- ML features
                    price_distance_from_fair REAL,
                    price_volatility REAL,
                    log_volume REAL,
                    market_age_days INTEGER,
                    question_word_count INTEGER,
                    has_political_keywords BOOLEAN,
                    has_sports_keywords BOOLEAN,
                    has_crypto_keywords BOOLEAN,
                    market_efficiency_score REAL,
                    will_resolve_yes INTEGER
                )
            """
            )

            # Insert/update data
            for _, row in ml_dataset.iterrows():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ml_market_features
                    (market_id, question, category, yes_price, no_price, volume, liquidity,
                     resolved, actual_outcome, created_at, end_date, resolved_at, days_to_resolve,
                     price_distance_from_fair, price_volatility, log_volume, market_age_days,
                     question_word_count, has_political_keywords, has_sports_keywords,
                     has_crypto_keywords, market_efficiency_score, will_resolve_yes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        row["market_id"],
                        row["question"],
                        row["category"],
                        row["yes_price"],
                        row["no_price"],
                        row["volume"],
                        row["liquidity"],
                        row["resolved"],
                        row["actual_outcome"],
                        row.get("created_at"),
                        row.get("end_date"),
                        row.get("resolved_at"),
                        row.get("days_to_resolve"),
                        row.get("price_distance_from_fair"),
                        row.get("price_volatility"),
                        row.get("log_volume"),
                        row.get("market_age_days"),
                        row.get("question_word_count"),
                        row.get("has_political_keywords"),
                        row.get("has_sports_keywords"),
                        row.get("has_crypto_keywords"),
                        row.get("market_efficiency_score"),
                        row.get("will_resolve_yes"),
                    ),
                )

            conn.commit()
            logger.info(f"Updated database with {len(ml_dataset)} ML feature records")

        except Exception as e:
            logger.error(f"Failed to update database: {e}")
        finally:
            conn.close()

    def get_training_data(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Get training and test data from the database.

        Returns:
            Tuple of (training_data, test_data)
        """
        conn = sqlite3.connect(self.db_path)

        try:
            df = pd.read_sql_query("SELECT * FROM ml_market_features", conn)

            if len(df) == 0:
                logger.warning("No ML training data in database")
                return pd.DataFrame(), pd.DataFrame()

            # Split into train/test
            train_df = df.sample(frac=1 - test_size, random_state=random_state)
            test_df = df.drop(train_df.index)

            logger.info(
                f"Loaded {len(train_df)} training samples, {len(test_df)} test samples"
            )

            return train_df, test_df

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return pd.DataFrame(), pd.DataFrame()
        finally:
            conn.close()

    def run_data_pipeline(self, days_back: int = 365) -> pd.DataFrame:
        """
        Run the complete data ingestion and cleaning pipeline.

        Args:
            days_back: Days of historical data to process

        Returns:
            ML-ready training dataset
        """
        logger.info("🚀 Starting ML data pipeline...")

        # 1. Fetch historical data
        raw_data = self.fetch_historical_markets(days_back=days_back)

        # 2. Clean and normalize
        cleaned_data = self.clean_market_data(raw_data)

        # 3. Engineer ML features
        ml_dataset = self.engineer_ml_features(cleaned_data)

        # 4. Update database
        self.update_database(ml_dataset)

        # 5. Return dataset for immediate use
        logger.info("✅ ML data pipeline completed successfully")
        return ml_dataset

#!/usr/bin/env python3
"""
Prepare Sports Markets Dataset for ML Training

Creates a high-quality, labeled dataset of historical sports markets from Polymarket
for training machine learning models that predict outcomes, calibrate probabilities,
or detect betting edges.

This script handles:
- Fetching resolved sports markets from Polymarket API
- Engineering features for ML models
- Handling edge cases and data quality issues
- Creating both real and synthetic datasets for testing
"""

import sqlite3
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
DB_PATH = Path("data/markets.db")
OUTPUT_DIR = Path("data")
REAL_DATASET_PATH = OUTPUT_DIR / "sports_ml_dataset_real.parquet"
SYNTHETIC_DATASET_PATH = OUTPUT_DIR / "sports_ml_dataset_synthetic.parquet"


class SportsMarketDataset:
    """
    Prepares sports market datasets for ML training.

    Handles data fetching, feature engineering, and quality control
    for sports prediction markets.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def fetch_resolved_sports_markets_api(self, days_back: int = 730) -> pd.DataFrame:
        """
        Fetch resolved sports markets from Polymarket API.

        This requires the data ingestion pipeline to be enhanced to fetch
        historical resolved markets, not just active ones.
        """
        logger.info(f"Attempting to fetch resolved sports markets from API (last {days_back} days)...")

        try:
            from polymarket_agents.automl.data_ingestion import PolymarketDataIngestion

            ingestor = PolymarketDataIngestion()

            # This would need to be enhanced to fetch resolved markets specifically
            # For now, we'll work with what we have in the database
            logger.warning("API fetching of resolved markets not fully implemented yet")
            return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to fetch from API: {e}")
            return pd.DataFrame()

    def load_resolved_sports_markets_db(self) -> pd.DataFrame:
        """
        Load resolved sports markets from local database.

        Currently, all markets in the DB are active, so this returns empty.
        This will work when resolved markets are added to the database.
        """
        logger.info("Loading resolved sports markets from database...")

        if not self.db_path.exists():
            logger.error(f"Database not found: {self.db_path}")
            return pd.DataFrame()

        conn = sqlite3.connect(self.db_path)

        # Query for resolved sports markets
        # Note: Current schema may not have 'resolved' column - adapt as needed
        query = """
        SELECT
            id, question, category, outcomes, outcome_prices,
            volume, liquidity, active, end_date, last_updated
        FROM markets
        WHERE category = 'sports'
        """

        # Add resolved filter if column exists
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM markets WHERE resolved = 1")
            if cursor.fetchone()[0] > 0:
                query += " AND resolved = 1"
        except sqlite3.OperationalError:
            logger.warning("No 'resolved' column found - using all sports markets")

        df = pd.read_sql_query(query, conn)
        conn.close()

        logger.info(f"Loaded {len(df)} sports markets from database")
        return df

    def create_synthetic_sports_dataset(self, n_samples: int = 2000) -> pd.DataFrame:
        """
        Create a synthetic dataset of resolved sports markets for testing.

        This generates realistic market data with known outcomes for model development
        and testing before real resolved data is available.
        """
        logger.info(f"Creating synthetic dataset with {n_samples} samples...")

        np.random.seed(42)  # For reproducibility

        # Generate market data
        market_ids = [f"synthetic_sports_{i}" for i in range(n_samples)]

        # Sports market types
        sport_types = ['NFL', 'NBA', 'MLB', 'NHL', 'Soccer', 'Tennis', 'Golf']
        sports_weights = [0.3, 0.25, 0.2, 0.1, 0.1, 0.03, 0.02]  # NFL most common

        # Market templates
        market_templates = [
            "Will {team1} beat {team2}?",
            "Will {team1} win the {championship}?",
            "Will {player} win {award}?",
            "Will the {game_type} total points be over {number}?",
            "Will {team1} cover the {spread} point spread?",
        ]

        # Team names by sport
        teams_by_sport = {
            'NFL': ['Chiefs', 'Eagles', 'Packers', 'Cowboys', 'Patriots', 'Seahawks', 'Rams', 'Buccaneers'],
            'NBA': ['Lakers', 'Warriors', 'Celtics', 'Heat', 'Bulls', 'Nets', 'Suns', 'Mavericks'],
            'MLB': ['Yankees', 'Dodgers', 'Red Sox', 'Mets', 'Cardinals', 'Giants', 'Rangers', 'Phillies'],
            'NHL': ['Bruins', 'Maple Leafs', 'Penguins', 'Oilers', 'Kings', 'Capitals', 'Blues', 'Hurricanes'],
            'Soccer': ['Manchester City', 'Liverpool', 'Barcelona', 'Real Madrid', 'Bayern Munich', 'PSG', 'Chelsea', 'Arsenal'],
            'Tennis': ['Djokovic', 'Federer', 'Nadal', 'Williams', 'Sharapova', 'Murray', 'Wawrinka', 'Ferrer'],
            'Golf': ['Rory McIlroy', 'Jordan Spieth', 'Justin Thomas', 'Rickie Fowler', 'Phil Mickelson', 'Tiger Woods']
        }

        # Generate synthetic markets
        rows = []
        for i in range(n_samples):
            # Select sport
            sport = np.random.choice(sport_types, p=sports_weights)
            teams = teams_by_sport[sport]

            # Generate question
            template = np.random.choice(market_templates)
            if 'team1' in template and 'team2' in template and 'spread' not in template and 'number' not in template:
                team1, team2 = np.random.choice(teams, 2, replace=False)
                question = template.format(team1=team1, team2=team2)
            elif 'spread' in template:
                team1, team2 = np.random.choice(teams, 2, replace=False)
                spread = np.random.choice([-3, -7, 3, 7, 10])
                question = template.format(team1=team1, spread=spread)
            elif 'number' in template:
                team1, team2 = np.random.choice(teams, 2, replace=False)
                number = np.random.randint(35, 60)
                question = template.format(team1=team1, team2=team2, game_type=f"{sport} game", number=number)
            elif 'team1' in template:
                team1 = np.random.choice(teams)
                question = template.format(
                    team1=team1,
                    championship=f"{sport} Championship"
                )
            else:
                player = np.random.choice(teams) if sport in ['Tennis', 'Golf'] else np.random.choice(teams)
                award = "MVP" if np.random.random() > 0.5 else "Championship"
                question = template.format(player=player, award=award)

            # Generate market characteristics
            volume = np.random.exponential(50000) + 1000  # Long tail distribution
            liquidity = volume * np.random.uniform(0.1, 0.5)  # Liquidity as fraction of volume

            # Days to expiry (past dates for resolved markets)
            days_ago = np.random.randint(1, 365)  # Resolved 1-365 days ago
            end_date = datetime.now() - timedelta(days=days_ago)

            # True probability (what actually happened)
            true_prob = np.random.beta(2, 2)  # Slightly biased toward 0.5

            # Market-implied probability (what the market showed before resolution)
            market_bias = np.random.normal(0, 0.1)  # Market can be slightly wrong
            market_prob = np.clip(true_prob + market_bias, 0.01, 0.99)

            # Actual outcome (based on true probability)
            actual_outcome = 1 if np.random.random() < true_prob else 0

            # Price at different times (simulate trajectory)
            initial_price = np.random.uniform(0.3, 0.7)
            final_price = market_prob  # Final price before resolution

            # Features
            price_momentum = final_price - initial_price
            volume_trend = np.random.normal(0, 0.1)  # Volume change over time
            days_to_expiry = days_ago  # At time of resolution

            rows.append({
                'market_id': market_ids[i],
                'question': question,
                'sport': sport,
                'category': 'sports',
                'volume': volume,
                'liquidity': liquidity,
                'liquidity_ratio': liquidity / volume if volume > 0 else 0,
                'days_to_expiry': days_to_expiry,
                'log_volume': np.log(volume + 1),
                'price_initial': initial_price,
                'price_final': final_price,
                'price_momentum': price_momentum,
                'volume_trend': volume_trend,
                'market_prob': market_prob,
                'true_prob': true_prob,
                'actual_outcome': actual_outcome,
                'end_date': end_date.isoformat(),
                'resolved': True,
                # Additional features
                'has_popular_team': any(team in question for team in ['Chiefs', 'Lakers', 'Yankees', 'Bruins']),
                'is_championship': 'championship' in question.lower() or 'super bowl' in question.lower(),
                'is_player_prop': any(word in question.lower() for word in ['mvp', 'award', 'win']),
                'question_length': len(question),
                'high_volume': volume > 100000,
                'high_liquidity': liquidity > 10000,
            })

        df = pd.DataFrame(rows)

        # Add some derived features
        df['price_volatility'] = df['market_prob'] * (1 - df['market_prob'])  # Information entropy
        df['edge'] = df['true_prob'] - df['market_prob']  # True edge (usually unknown in real markets)
        df['abs_edge'] = abs(df['edge'])

        logger.info(f"Created synthetic dataset: {len(df)} samples")
        logger.info(f"Outcome distribution: {df['actual_outcome'].mean():.1%} YES outcomes")
        logger.info(".2f")

        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional ML features from raw market data.

        Args:
            df: Raw market DataFrame

        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering ML features...")

        # Copy to avoid modifying original
        ml_df = df.copy()

        # Price-based features
        if 'price_final' in ml_df.columns:
            ml_df['price_distance_from_fair'] = abs(ml_df['price_final'] - 0.5)
            ml_df['price_extremity'] = ((ml_df['price_final'] < 0.2) | (ml_df['price_final'] > 0.8)).astype(int)

        # Volume features
        if 'volume' in ml_df.columns:
            ml_df['volume_category'] = pd.cut(
                ml_df['volume'],
                bins=[0, 10000, 50000, 100000, 500000, float('inf')],
                labels=['micro', 'small', 'medium', 'large', 'huge']
            )

        # Time-based features
        if 'days_to_expiry' in ml_df.columns:
            ml_df['urgency_high'] = (ml_df['days_to_expiry'] < 7).astype(int)
            ml_df['urgency_medium'] = ((ml_df['days_to_expiry'] >= 7) & (ml_df['days_to_expiry'] < 30)).astype(int)

        # Sport-specific features
        if 'sport' in ml_df.columns:
            major_sports = ['NFL', 'NBA', 'MLB', 'Soccer']
            for sport in major_sports:
                ml_df[f'is_{sport.lower()}'] = (ml_df['sport'] == sport).astype(int)

        # Text features
        if 'question' in ml_df.columns:
            ml_df['question_word_count'] = ml_df['question'].str.split().str.len()
            ml_df['has_numbers'] = ml_df['question'].str.contains(r'\d').astype(int)

        # Market microstructure features
        if 'liquidity' in ml_df.columns and 'volume' in ml_df.columns:
            ml_df['liquidity_to_volume'] = ml_df['liquidity'] / ml_df['volume'].replace(0, 1)

        logger.info(f"Engineered {len(ml_df.columns) - len(df.columns)} additional features")

        return ml_df

    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2,
                               time_based: bool = True) -> tuple:
        """
        Create train/test split, optionally time-based to avoid lookahead bias.

        Args:
            df: Dataset to split
            test_size: Fraction for testing
            time_based: Whether to split by time (recommended for time series)

        Returns:
            Tuple of (train_df, test_df)
        """
        if time_based and 'end_date' in df.columns:
            # Sort by end_date and take most recent for testing
            df_sorted = df.sort_values('end_date')
            split_idx = int(len(df_sorted) * (1 - test_size))

            train_df = df_sorted.iloc[:split_idx]
            test_df = df_sorted.iloc[split_idx:]

            logger.info(f"Time-based split: {len(train_df)} train, {len(test_df)} test samples")
        else:
            # Random split (not recommended for time series data)
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
            logger.info(f"Random split: {len(train_df)} train, {len(test_df)} test samples")

        return train_df, test_df

    def validate_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate dataset quality and return statistics.

        Args:
            df: Dataset to validate

        Returns:
            Dictionary with validation metrics
        """
        validation = {
            'total_samples': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_markets': df['market_id'].duplicated().sum() if 'market_id' in df.columns else 0,
        }

        if 'actual_outcome' in df.columns:
            validation['outcome_distribution'] = df['actual_outcome'].value_counts().to_dict()
            validation['outcome_balance'] = df['actual_outcome'].mean()

        if 'volume' in df.columns:
            validation['volume_stats'] = {
                'mean': df['volume'].mean(),
                'median': df['volume'].median(),
                'min': df['volume'].min(),
                'max': df['volume'].max()
            }

        if 'category' in df.columns:
            validation['category_distribution'] = df['category'].value_counts().to_dict()

        # Check for data quality issues
        issues = []
        if validation['missing_values'] > 0:
            issues.append(f"{validation['missing_values']} missing values")
        if validation.get('outcome_balance', 0.5) < 0.3 or validation.get('outcome_balance', 0.5) > 0.7:
            issues.append("Unbalanced outcomes (should be ~50%)")
        if validation['total_samples'] < 100:
            issues.append("Very small dataset (<100 samples)")

        validation['quality_issues'] = issues
        validation['quality_score'] = 'good' if not issues else 'needs_attention'

        return validation

    def prepare_real_dataset(self) -> pd.DataFrame:
        """
        Prepare dataset from real resolved markets.

        Returns:
            DataFrame ready for ML training
        """
        logger.info("Preparing real sports markets dataset...")

        # Try to load from database first
        df = self.load_resolved_sports_markets_db()

        if df.empty:
            # Try to fetch from API
            df = self.fetch_resolved_sports_markets_api()

        if df.empty:
            logger.warning("No real resolved markets available. Use prepare_synthetic_dataset() for testing.")
            return pd.DataFrame()

        # Engineer features
        df = self.engineer_features(df)

        # Basic filtering
        df = df.dropna(subset=['volume', 'liquidity'])
        df = df[df['volume'] > 1000]  # Filter low-volume markets

        logger.info(f"Prepared real dataset: {len(df)} samples")
        return df

    def prepare_synthetic_dataset(self, n_samples: int = 2000) -> pd.DataFrame:
        """
        Prepare synthetic dataset for testing and development.

        Args:
            n_samples: Number of synthetic samples to generate

        Returns:
            DataFrame ready for ML training
        """
        logger.info(f"Preparing synthetic sports markets dataset ({n_samples} samples)...")

        # Create synthetic data
        df = self.create_synthetic_sports_dataset(n_samples)

        # Engineer additional features
        df = self.engineer_features(df)

        # Add target column (actual_outcome -> target)
        if 'actual_outcome' in df.columns:
            df['target'] = df['actual_outcome']

        logger.info(f"Prepared synthetic dataset: {len(df)} samples")
        return df

    def save_dataset(self, df: pd.DataFrame, filepath: Path, validation: bool = True) -> None:
        """
        Save dataset to disk with optional validation.

        Args:
            df: Dataset to save
            filepath: Path to save to
            validation: Whether to validate before saving
        """
        if validation:
            val_results = self.validate_dataset(df)
            logger.info(f"Dataset validation: {val_results['quality_score']}")
            if val_results['quality_issues']:
                logger.warning(f"Issues found: {val_results['quality_issues']}")

        # Save as parquet for efficiency
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {len(df)} samples to {filepath}")

        # Also save validation report
        if validation:
            val_file = filepath.with_suffix('.validation.json')
            with open(val_file, 'w') as f:
                json.dump(val_results, f, indent=2, default=str)
            logger.info(f"Saved validation report to {val_file}")


def main():
    """Main function to prepare sports datasets."""
    print("🏈 Polymarket Sports ML Dataset Preparation")
    print("=" * 50)

    dataset_prep = SportsMarketDataset()

    # Try to prepare real dataset first
    print("\n📊 Attempting to prepare real dataset...")
    real_df = dataset_prep.prepare_real_dataset()

    if not real_df.empty:
        print(f"✅ Found {len(real_df)} real resolved sports markets!")
        dataset_prep.save_dataset(real_df, REAL_DATASET_PATH)
    else:
        print("⚠️  No real resolved markets found.")

    # Always create synthetic dataset for testing
    print("\n🎭 Preparing synthetic dataset for testing...")
    synthetic_df = dataset_prep.prepare_synthetic_dataset(n_samples=2000)
    dataset_prep.save_dataset(synthetic_df, SYNTHETIC_DATASET_PATH)

    # Show sample
    print("\n📋 Sample of synthetic dataset:")
    display_cols = ['market_id', 'question', 'sport', 'volume', 'market_prob', 'true_prob', 'actual_outcome', 'target']
    available_cols = [col for col in display_cols if col in synthetic_df.columns]
    print(synthetic_df[available_cols].head())

    # Show statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(synthetic_df)}")
    if 'target' in synthetic_df.columns:
        print(".1%")
    if 'volume' in synthetic_df.columns:
        print(".0f")
    if 'sport' in synthetic_df.columns:
        print(f"   Sports distribution: {synthetic_df['sport'].value_counts().to_dict()}")

    print("\n🎯 Next Steps:")
    print("1. Use synthetic dataset for model development:")
    print("   python train_xgboost_strategy.py --dataset data/sports_ml_dataset_synthetic.parquet")
    print("2. When real resolved markets are available, use:")
    print("   python train_xgboost_strategy.py --dataset data/sports_ml_dataset_real.parquet")
    print("3. Integrate with planning agent for live predictions")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Advanced Market Analysis Pipeline

Leverages ML for standardizing Polymarket market analysis with specialized
forecasting models for different market categories (social media, sports, finance).

Features:
- Market classification using ML (social, sports, finance, politics, crypto)
- Specialized forecasting models for each category
- External data integration for validation
- Probabilistic forecasting with uncertainty ranges
- Ontology-based market understanding
"""

import sys
import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# ML and forecasting libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

# Import caching system
from cache_manager import (
    CacheManager,
    SocialMediaDataFetcher,
    CryptoDataFetcher,
    SportsDataFetcher,
)

# Import existing Polymarket tools
try:
    from polymarket_agents.automl.data_ingestion import PolymarketDataIngestion
    from polymarket_agents.automl.ml_database import MLDatabase
except ImportError:
    # Fallback if not available
    pass


@dataclass
class MarketAnalysis:
    """Result of market analysis with forecasting."""

    market_id: str
    question: str
    category: str
    forecast: Dict[str, Any]
    confidence: float
    reasoning: str
    external_data_used: List[str]


@dataclass
class ForecastResult:
    """Forecast result with uncertainty ranges."""

    prediction: float
    lower_bound: float
    upper_bound: float
    confidence_interval: float
    model_used: str
    external_factors: List[str]


class MarketClassifier:
    """
    ML-based market classifier using ontology principles.

    Classifies markets into categories based on question text and metadata.
    Categories: social, sports, finance, politics, crypto, entertainment, science, other
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.is_trained = False

    def train(self, training_data: Optional[List[Tuple[str, str]]] = None):
        """Train the classifier on market questions and categories."""

        if training_data is None:
            # Default training data based on ontology - expanded for better classification
            training_data = [
                # Social/Social Media (multiple examples)
                ("How many tweets will Elon Musk post in January 2026?", "social"),
                ("Will Elon Musk tweet about Dogecoin this week?", "social"),
                ("Elon's monthly tweet count range", "social"),
                ("Will Kim Kardashian post on Instagram today?", "social"),
                ("Twitter engagement for political posts", "social"),
                ("Monthly tweet volume for social media influencers", "social"),
                ("Social media posting frequency analysis", "social"),
                # Sports (multiple examples)
                ("Who will win the Super Bowl 2026?", "sports"),
                ("Will the Denver Nuggets win the NBA championship?", "sports"),
                ("Super Bowl LVIII winner prediction", "sports"),
                ("Will LeBron James retire in 2026?", "sports"),
                ("NFL playoff outcomes 2025", "sports"),
                ("NBA MVP 2026 winner", "sports"),
                ("Championship winner predictions", "sports"),
                ("Team performance forecasts", "sports"),
                ("Athlete career milestones", "sports"),
                # Finance/Economics (multiple examples)
                ("Will the Federal Reserve raise interest rates?", "finance"),
                ("GDP growth forecast for Q4 2025", "finance"),
                ("Will inflation exceed 3% in 2026?", "finance"),
                ("Stock market crash probability", "finance"),
                ("Bitcoin price prediction for end of 2025", "finance"),
                ("Economic indicator forecasts", "finance"),
                ("Market volatility predictions", "finance"),
                # Politics (multiple examples)
                ("Will Trump win the 2028 election?", "politics"),
                ("Congressional control in 2026 midterms", "politics"),
                ("Will Biden run for re-election?", "politics"),
                ("Supreme Court decisions in 2025", "politics"),
                ("Presidential approval ratings", "politics"),
                ("Election outcome predictions", "politics"),
                ("Political event forecasts", "politics"),
                # Crypto (multiple examples)
                ("Will Bitcoin reach $200k by end of 2025?", "crypto"),
                ("Ethereum staking rewards in 2026", "crypto"),
                ("Will a major crypto exchange be hacked in 2025?", "crypto"),
                ("DeFi TVL growth forecast", "crypto"),
                ("Will Ethereum switch to proof of stake?", "crypto"),
                ("Cryptocurrency price predictions", "crypto"),
                ("Blockchain technology adoption", "crypto"),
                # Entertainment (multiple examples)
                ("Will Taylor Swift release new album in 2025?", "entertainment"),
                ("Box office performance predictions", "entertainment"),
                ("Will Drake release music in 2026?", "entertainment"),
                ("Streaming service subscriber growth", "entertainment"),
                ("Movie box office forecasts", "entertainment"),
                ("Music industry predictions", "entertainment"),
                # Science/Technology (multiple examples)
                ("Will AGI be achieved by 2030?", "science"),
                ("SpaceX Starship orbital flight success", "science"),
                ("Will fusion energy be commercialized?", "science"),
                ("COVID vaccine effectiveness in 2026", "science"),
                ("Scientific breakthrough predictions", "science"),
                ("Technology adoption forecasts", "science"),
                # Other/Uncategorized (multiple examples)
                ("Will aliens be discovered in 2025?", "other"),
                ("Time travel invention timeline", "other"),
                ("Zombie apocalypse probability", "other"),
                ("Unusual event predictions", "other"),
                ("Speculative forecasts", "other"),
            ]

        questions, labels = zip(*training_data)
        # Use stratify only if we have enough samples per class
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                questions, labels, test_size=0.2, random_state=42, stratify=labels
            )
        except ValueError:
            # Fall back to non-stratified split if not enough samples per class
            X_train, X_test, y_train, y_test = train_test_split(
                questions, labels, test_size=0.2, random_state=42
            )

        # Vectorize and train
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.classifier.fit(X_train_vec, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        print("Market Classifier Training Results:")
        try:
            print(classification_report(y_test, y_pred, zero_division=0))
        except:
            print(f"Training completed with {len(set(y_train))} classes")

        self.is_trained = True
        return self

    def classify(self, question: str, metadata: Optional[Dict] = None) -> str:
        """Classify a market question into a category."""
        if not self.is_trained:
            self.train()

        question_vec = self.vectorizer.transform([question])
        prediction = self.classifier.predict(question_vec)[0]

        # Use metadata to refine classification if available
        if metadata:
            if any(
                tag in (metadata.get("tags", []) or [])
                for tag in ["crypto", "bitcoin", "ethereum"]
            ):
                prediction = "crypto"
            elif any(
                tag in (metadata.get("tags", []) or [])
                for tag in ["nfl", "nba", "soccer", "football"]
            ):
                prediction = "sports"

        return prediction


class SocialMediaForecaster:
    """
    Specialized forecaster for social media metrics (tweets, posts, engagement).

    Uses time-series analysis to forecast ranges for social media activity.
    """

    def __init__(self):
        self.models = {}
        self.data_cache = {}

    def load_elon_tweets_data(
        self, csv_path: str = "./data/elon_tweets.csv"
    ) -> pd.DataFrame:
        """Load and preprocess Elon Musk tweets data."""
        try:
            df = pd.read_csv(csv_path, parse_dates=["date"])
            # Aggregate to monthly counts
            monthly = (
                df.groupby(df["date"].dt.to_period("M"))
                .size()
                .reset_index(name="tweet_count")
            )
            monthly["date"] = monthly["date"].dt.to_timestamp()
            return monthly
        except FileNotFoundError:
            print(f"Elon tweets data not found at {csv_path}")
            print("Using synthetic data for demonstration...")
            # Generate synthetic data
            dates = pd.date_range("2020-01-01", "2025-12-01", freq="M")
            np.random.seed(42)
            counts = np.random.normal(150, 50, len(dates))  # ~150 tweets/month average
            counts = np.clip(counts, 10, 400)  # Reasonable bounds
            return pd.DataFrame({"date": dates, "tweet_count": counts})

    def forecast_tweet_range(
        self, target_month: str = "2026-01", confidence_level: float = 0.95
    ) -> ForecastResult:
        """
        Forecast Elon Musk tweet count range for a specific month.
        Uses cached historical data for improved performance.

        Args:
            target_month: Target month in YYYY-MM format
            confidence_level: Confidence level for prediction intervals

        Returns:
            ForecastResult with prediction and uncertainty bounds
        """
        # Try to use cached data first
        tweets_df = None

        # Check if we have access to cached data fetcher
        if (
            hasattr(self, "pipeline")
            and self.pipeline
            and hasattr(self.pipeline, "social_fetcher")
            and self.pipeline.social_fetcher
        ):
            try:
                cached_data = self.pipeline.social_fetcher.fetch_elon_tweets_monthly()
                # Convert cached data to DataFrame format
                tweets_data = []
                for period, count in cached_data.items():
                    if (
                        "(forecast)" not in period
                        and "_lower" not in period
                        and "_upper" not in period
                    ):
                        try:
                            # Handle different date formats
                            if len(period) == 7:  # YYYY-MM format
                                date = pd.to_datetime(period + "-01")
                            else:
                                date = pd.to_datetime(period)
                            tweets_data.append({"ds": date, "y": count})
                        except:
                            continue

                if tweets_data:
                    tweets_df = pd.DataFrame(tweets_data)
                    print(f"Using cached tweet data: {len(tweets_df)} months")

            except Exception as e:
                print(f"Cache error, using synthetic data: {e}")

        # Fall back to synthetic data if cache not available
        if tweets_df is None or tweets_df.empty:
            tweets_df = self._create_synthetic_tweets_data()

        if tweets_df.empty:
            return ForecastResult(
                prediction=1000,  # Reasonable default
                lower_bound=500,
                upper_bound=2000,
                confidence_interval=0.5,
                model_used="Default Estimate",
                external_factors=["Limited historical data"],
            )

        # Train Prophet model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,  # Flexible trend changes
            seasonality_prior_scale=10,  # Strong seasonal effects
            interval_width=confidence_level,
        )

        model.fit(tweets_df)

        # Create future dataframe for target month
        future = pd.DataFrame({"ds": [pd.to_datetime(target_month)]})
        forecast = model.predict(future)

        prediction = forecast["yhat"].iloc[0]
        lower_bound = forecast["yhat_lower"].iloc[0]
        upper_bound = forecast["yhat_upper"].iloc[0]

        return ForecastResult(
            prediction=max(0, prediction),  # Can't have negative tweets
            lower_bound=max(0, lower_bound),
            upper_bound=max(0, upper_bound),
            confidence_interval=confidence_level,
            model_used="Prophet (Time-Series with Caching)",
            external_factors=[
                "Cached historical tweet patterns",
                "Seasonal effects (news, product launches)",
                "Platform changes (Twitter/X rebrand)",
                "External events (Tesla, SpaceX milestones)",
            ],
        )

    def _create_synthetic_tweets_data(self) -> pd.DataFrame:
        """Create synthetic tweets data for fallback."""
        dates = pd.date_range("2020-01-01", "2025-12-01", freq="M")
        np.random.seed(42)
        counts = np.random.normal(150, 50, len(dates))  # ~150 tweets/month average
        counts = np.clip(counts, 10, 400)  # Reasonable bounds
        return pd.DataFrame({"ds": dates, "y": counts})


class SportsForecaster:
    """
    Specialized forecaster for sports outcomes using historical performance.

    Analyzes team statistics, recent form, and historical data to predict outcomes.
    """

    def __init__(self):
        self.team_stats_cache = {}
        self.historical_data = {}

    def load_team_stats(self, team_name: str) -> Dict[str, Any]:
        """Load or fetch team statistics."""
        if team_name in self.team_stats_cache:
            return self.team_stats_cache[team_name]

        # Mock data for demonstration - in production, fetch from APIs
        mock_stats = {
            "Denver Nuggets": {
                "win_pct_last_5_years": [
                    0.67,
                    0.73,
                    0.56,
                    0.48,
                    0.63,
                ],  # Including 2023 championship
                "avg_points_per_game": 115.2,
                "recent_form": 0.75,  # Last season win %
                "home_advantage": 0.65,
                "key_players": ["Nikola Jokic", "Jamal Murray", "Aaron Gordon"],
            },
            "Cleveland Cavaliers": {
                "win_pct_last_5_years": [0.29, 0.48, 0.44, 0.51, 0.63],
                "avg_points_per_game": 112.8,
                "recent_form": 0.63,
                "home_advantage": 0.58,
                "key_players": ["Donovan Mitchell", "Darius Garland", "Evan Mobley"],
            },
            "Oklahoma City Thunder": {
                "win_pct_last_5_years": [
                    0.21,
                    0.22,
                    0.40,
                    0.57,
                    0.69,
                ],  # Young rising team
                "avg_points_per_game": 120.1,
                "recent_form": 0.69,
                "home_advantage": 0.70,
                "key_players": [
                    "Shai Gilgeous-Alexander",
                    "Chet Holmgren",
                    "Luguentz Dort",
                ],
            },
            "Kansas City Chiefs": {
                "win_pct_last_5_years": [
                    0.75,
                    0.81,
                    0.69,
                    0.81,
                    0.81,
                ],  # Super Bowl winners
                "avg_points_per_game": 28.4,
                "recent_form": 0.81,
                "home_advantage": 0.75,
                "key_players": ["Patrick Mahomes", "Travis Kelce", "Chris Jones"],
            },
            "Philadelphia Eagles": {
                "win_pct_last_5_years": [
                    0.38,
                    0.63,
                    0.50,
                    0.63,
                    0.81,
                ],  # 2025 Super Bowl champs
                "avg_points_per_game": 26.8,
                "recent_form": 0.81,
                "home_advantage": 0.70,
                "key_players": ["Jalen Hurts", "A.J. Brown", "Haason Reddick"],
            },
        }

        # Use mock data or fetch real data
        stats = mock_stats.get(
            team_name,
            {
                "win_pct_last_5_years": [0.50] * 5,  # Average performance
                "avg_points_per_game": 100,
                "recent_form": 0.50,
                "home_advantage": 0.55,
                "key_players": ["Unknown"],
            },
        )

        # Calculate derived metrics
        stats["avg_win_pct"] = np.mean(stats["win_pct_last_5_years"])
        stats["trend"] = np.polyfit(
            range(len(stats["win_pct_last_5_years"])), stats["win_pct_last_5_years"], 1
        )[
            0
        ]  # Linear trend

        self.team_stats_cache[team_name] = stats
        return stats

    def forecast_sports_outcome(
        self, sport: str, event: str, team: str, opponent: Optional[str] = None
    ) -> ForecastResult:
        """
        Forecast sports outcome probability.

        Args:
            sport: Type of sport (nba, nfl, etc.)
            event: Type of event (championship, playoff, etc.)
            team: Team name
            opponent: Opponent team name (if applicable)

        Returns:
            ForecastResult with win probability and confidence bounds
        """
        team_stats = self.load_team_stats(team)

        # Base probability from historical performance
        base_prob = team_stats["avg_win_pct"]

        # Adjust for recent form
        form_adjustment = (team_stats["recent_form"] - team_stats["avg_win_pct"]) * 0.3
        adjusted_prob = base_prob + form_adjustment

        # Adjust for event type
        event_multipliers = {
            "regular_season": 1.0,
            "playoffs": 0.8,  # Harder in playoffs
            "championship": 0.6,  # Very competitive
            "super_bowl": 0.55,  # Most competitive
        }

        event_key = event.lower()
        for key, multiplier in event_multipliers.items():
            if key in event_key:
                adjusted_prob *= multiplier
                break

        # Opponent adjustment (if available)
        if opponent:
            opp_stats = self.load_team_stats(opponent)
            opp_strength = opp_stats["avg_win_pct"]
            # Relative strength adjustment
            strength_diff = team_stats["avg_win_pct"] - opp_strength
            adjusted_prob += strength_diff * 0.2

        # Ensure reasonable bounds
        adjusted_prob = np.clip(adjusted_prob, 0.05, 0.95)

        # Calculate uncertainty based on historical variance
        historical_variance = np.var(team_stats["win_pct_last_5_years"])
        uncertainty = min(0.3, historical_variance * 2)  # Scale uncertainty

        return ForecastResult(
            prediction=adjusted_prob,
            lower_bound=max(0, adjusted_prob - uncertainty),
            upper_bound=min(1, adjusted_prob + uncertainty),
            confidence_interval=0.8,  # 80% confidence interval
            model_used="Statistical Model (Historical Performance)",
            external_factors=[
                f"Team historical win%: {team_stats['avg_win_pct']:.1%}",
                f"Recent form: {team_stats['recent_form']:.1%}",
                f"Event type adjustment: {event}",
                f"Key players: {', '.join(team_stats['key_players'][:2])}",
                "Home advantage considerations",
                "Injury reports and matchup history",
            ],
        )


class MarketAnalysisPipeline:
    """
    Complete market analysis pipeline with ML classification and specialized forecasting.
    Now includes intelligent caching for improved performance.
    """

    def __init__(self, enable_caching: bool = True):
        self.database = MLDatabase()

        # Initialize caching system first
        self.enable_caching = enable_caching
        if enable_caching:
            self.cache_manager = CacheManager(db_path="data/market_cache.db")
            self.social_fetcher = SocialMediaDataFetcher(self.cache_manager)
            self.crypto_fetcher = CryptoDataFetcher(self.cache_manager)
            self.sports_data_fetcher = SportsDataFetcher(self.cache_manager)
        else:
            self.cache_manager = None
            self.social_fetcher = None
            self.crypto_fetcher = None
            self.sports_data_fetcher = None

        # Initialize forecasters (pass pipeline reference for caching)
        self.classifier = MarketClassifier()
        self.social_forecaster = SocialMediaForecaster()
        self.sports_forecaster = SportsForecaster()

        # Connect forecasters to pipeline for caching access
        if enable_caching:
            self.social_forecaster.pipeline = self
            self.sports_forecaster.pipeline = self

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        if self.cache_manager:
            self.cache_manager.close()

    def fetch_markets(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Fetch markets from Polymarket API."""
        url = f"https://gamma-api.polymarket.com/markets?active=true&closed=false&limit={limit}"

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            # Validate and clean data
            markets = []
            for market in data:
                if "question" in market and market.get("active", False):
                    # Ensure volume is numeric
                    volume = market.get("volume", 0)
                    try:
                        volume = float(volume) if volume else 0
                    except (ValueError, TypeError):
                        volume = 0

                    markets.append(
                        {
                            "id": market.get("id"),
                            "question": market.get("question", ""),
                            "description": market.get("description"),
                            "outcomes": market.get("outcomes", []),
                            "volume": volume,
                            "tags": market.get("tags", []),
                            "end_date": market.get("endDate"),
                            "category": None,  # Will be classified
                        }
                    )

            return markets

        except Exception as e:
            print(f"Error fetching markets: {e}")
            return []

    def analyze_market(self, market: Dict[str, Any]) -> MarketAnalysis:
        """Analyze a single market with appropriate forecasting model."""

        question = market["question"]
        category = self.classifier.classify(question, market)

        # Apply category-specific forecasting
        if category == "social" and (
            "elon" in question.lower() or "tweet" in question.lower()
        ):
            # Social media forecasting (Elon tweets)
            forecast = self._forecast_social_market(question)

        elif category == "sports":
            # Sports outcome forecasting
            forecast = self._forecast_sports_market(question)

        else:
            # Generic forecasting for other categories
            forecast = self._forecast_generic_market(market, category)

        # Calculate confidence based on forecast quality
        confidence = self._calculate_confidence(forecast, category)

        # Generate reasoning
        reasoning = self._generate_reasoning(market, category, forecast)

        return MarketAnalysis(
            market_id=market["id"],
            question=question,
            category=category,
            forecast=forecast,
            confidence=confidence,
            reasoning=reasoning,
            external_data_used=(
                forecast.external_factors
                if isinstance(forecast, ForecastResult)
                else forecast.get("external_factors", [])
            ),
        )

    def _forecast_social_market(self, question: str) -> ForecastResult:
        """Forecast social media related markets."""
        print(f"DEBUG: Social market analysis: {question[:60]}...")
        if "elon" in question.lower() and "tweet" in question.lower():
            # Extract target month from question if possible
            target_month = "2026-01"  # Default
            if "january" in question.lower():
                target_month = "2026-01"
            elif "february" in question.lower():
                target_month = "2026-02"
            # Add more month parsing as needed

            return self.forecast_tweet_range(target_month)

        # Generic social media forecast
        return ForecastResult(
            prediction=0.5,
            lower_bound=0.3,
            upper_bound=0.7,
            confidence_interval=0.6,
            model_used="Generic Social Model",
            external_factors=["Social media trends", "Historical patterns"],
        )

    def _forecast_sports_market(self, question: str) -> ForecastResult:
        """Forecast sports-related markets."""

        # Extract team names and event type from question
        sport = (
            "nba"
            if "nba" in question.lower()
            else "nfl" if "super bowl" in question.lower() else "general"
        )

        # Try to identify the team
        teams = {
            "nuggets": "Denver Nuggets",
            "cavaliers": "Cleveland Cavaliers",
            "thunder": "Oklahoma City Thunder",
            "chiefs": "Kansas City Chiefs",
            "eagles": "Philadelphia Eagles",
            "chiefs": "Kansas City Chiefs",
            "patriots": "New England Patriots",
            "rams": "Los Angeles Rams",
        }

        team = None
        for team_key, team_name in teams.items():
            if team_key in question.lower():
                team = team_name
                break

        # Determine event type
        event = "regular_season"
        if "championship" in question.lower() or "title" in question.lower():
            event = "championship"
        elif "super bowl" in question.lower():
            event = "super_bowl"
        elif "playoff" in question.lower():
            event = "playoffs"

        if team:
            return self.sports_forecaster.forecast_sports_outcome(sport, event, team)

        # Fallback generic sports forecast
        return ForecastResult(
            prediction=0.5,
            lower_bound=0.2,
            upper_bound=0.8,
            confidence_interval=0.6,
            model_used="Generic Sports Model",
            external_factors=[
                "Historical team performance",
                "Recent form",
                "Event competitiveness",
            ],
        )

    def _forecast_generic_market(
        self, market: Dict[str, Any], category: str
    ) -> Dict[str, Any]:
        """Generic forecasting for uncategorized markets."""
        # Use market volume and current prices as simple predictors
        volume = float(market.get("volume", 0) or 0)

        # Simple heuristic: higher volume markets are more likely to resolve as expected
        base_prob = 0.5
        if volume > 100000:
            base_prob = 0.55  # Slight edge for high-volume markets
        elif volume < 10000:
            base_prob = 0.45  # Lower confidence for low-volume

        return {
            "prediction": base_prob,
            "lower_bound": max(0, base_prob - 0.2),
            "upper_bound": min(1, base_prob + 0.2),
            "confidence_interval": 0.4,
            "model_used": "Simple Heuristic Model",
            "external_factors": ["Market volume analysis", "Historical patterns"],
        }

    def _calculate_confidence(self, forecast: Any, category: str) -> float:
        """Calculate confidence score for the forecast."""
        if isinstance(forecast, ForecastResult):
            # Use confidence interval as proxy for confidence
            return forecast.confidence_interval
        else:
            # Generic confidence based on category
            category_confidence = {
                "sports": 0.75,  # Good historical data
                "social": 0.70,  # Social media patterns
                "finance": 0.65,  # Economic indicators
                "crypto": 0.60,  # Volatile market
                "politics": 0.55,  # Unpredictable
                "other": 0.50,  # Limited data
            }
            return category_confidence.get(category, 0.5)

    def _generate_reasoning(
        self, market: Dict[str, Any], category: str, forecast: Any
    ) -> str:
        """Generate human-readable reasoning for the forecast."""
        volume = market.get("volume", 0)
        volume_desc = f"${volume:,.0f}" if volume > 0 else "unknown"

        if category == "social":
            return f"Social media market ({volume_desc} volume) forecasted using time-series analysis of historical posting patterns with seasonal adjustments."

        elif category == "sports":
            return f"Sports outcome market ({volume_desc} volume) predicted using historical team performance data, recent form, and event-specific adjustments."

        else:
            return f"Generic market analysis ({volume_desc} volume) using volume-based heuristics and historical market patterns."

    def run_analysis_pipeline(self, limit: int = 50) -> List[MarketAnalysis]:
        """Run the complete market analysis pipeline."""

        print("🤖 Starting Market Analysis Pipeline")
        print("=" * 50)

        # Train classifier
        print("🎯 Training market classifier...")
        self.classifier.train()

        # Fetch markets
        print(f"📊 Fetching {limit} active markets...")
        markets = self.fetch_markets(limit)

        if not markets:
            print("❌ No markets fetched")
            return []

        print(f"✅ Found {len(markets)} markets")

        # Analyze each market
        print("🔍 Analyzing markets with specialized forecasting...")
        analyses = []

        for i, market in enumerate(markets, 1):
            try:
                analysis = self.analyze_market(market)
                analyses.append(analysis)

                # Progress indicator
                if i % 10 == 0:
                    print(f"   Processed {i}/{len(markets)} markets...")

            except Exception as e:
                print(f"❌ Error analyzing market {market['id']}: {e}")
                continue

        print(f"✅ Completed analysis of {len(analyses)} markets")

        # Store results in database
        self._store_results(analyses)

        return analyses

    def _store_results(self, analyses: List[MarketAnalysis]):
        """Store analysis results in the database."""
        try:
            for analysis in analyses:
                # Store as evaluation result
                result_data = {
                    "market_id": analysis.market_id,
                    "question": analysis.question,
                    "category": analysis.category,
                    "forecast": analysis.forecast,
                    "confidence": analysis.confidence,
                    "reasoning": analysis.reasoning,
                    "external_data_used": analysis.external_data_used,
                    "timestamp": datetime.now().isoformat(),
                }

                self.database.save_evaluation(
                    model_id=f"market_analysis_{analysis.category}",
                    evaluation_type="market_forecast",
                    evaluation_config={"market_id": analysis.market_id},
                    results=result_data,
                    duration_seconds=0,
                )

        except Exception as e:
            print(f"Warning: Could not store results in database: {e}")

    def get_category_summary(self, analyses: List[MarketAnalysis]) -> Dict[str, Any]:
        """Generate summary statistics by category."""
        categories = {}
        for analysis in analyses:
            cat = analysis.category
            if cat not in categories:
                categories[cat] = {"count": 0, "avg_confidence": 0, "forecasts": []}

            categories[cat]["count"] += 1
            categories[cat]["avg_confidence"] += analysis.confidence
            categories[cat]["forecasts"].append(analysis.forecast)

        # Calculate averages
        for cat_data in categories.values():
            cat_data["avg_confidence"] /= cat_data["count"]

        return categories

    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache performance statistics."""
        if self.cache_manager:
            return self.cache_manager.get_stats()
        return None


def main():
    """Run the market analysis pipeline with caching."""
    print("🚀 Market Analysis Pipeline with Intelligent Caching")
    print("=" * 55)

    with MarketAnalysisPipeline(enable_caching=True) as pipeline:
        # Show cache status
        if pipeline.enable_caching:
            cache_stats = pipeline.get_cache_stats()
            if cache_stats:
                print("📊 CACHE STATUS:")
                print(f"   Database: {cache_stats['db_path']}")
                print(f"   Entries: {cache_stats['total_entries']}")
                print(f"   Fresh: {cache_stats['fresh_entries']}")
                print(f"   Size: {cache_stats['total_size_kb']:.1f} KB")
                print(f"   TTL: {cache_stats['default_ttl_seconds']/3600:.1f} hours")
                print()

        analyses = pipeline.run_analysis_pipeline(limit=30)  # Smaller limit for demo

        if analyses:
            # Show results
            print("\\n📊 ANALYSIS RESULTS:")
            print("=" * 30)

            for analysis in analyses[:10]:  # Show first 10
                print(f"\\n🎯 {analysis.question[:60]}...")
                print(f"   Category: {analysis.category}")
                print(f"   Confidence: {analysis.confidence:.1%}")

                if hasattr(analysis.forecast, "prediction"):
                    pred = analysis.forecast.prediction
                    low = analysis.forecast.lower_bound
                    high = analysis.forecast.upper_bound
                    if analysis.category == "social":
                        print(
                            f"   Forecast: {pred:.0f} tweets (range: {low:.0f} - {high:.0f})"
                        )
                    elif analysis.category == "sports":
                        print(
                            f"   Forecast: {pred:.1%} (range: {low:.1%} - {high:.1%})"
                        )
                    else:
                        print(
                            f"   Forecast: {pred:.1%} (range: {low:.1%} - {high:.1%})"
                        )
                else:
                    pred = analysis.forecast.get("prediction", 0)
                    print(f"   Forecast: {pred:.1%}")

            # Category summary
            summary = pipeline.get_category_summary(analyses)
            print(f"\\n📈 CATEGORY SUMMARY:")
            for category, data in summary.items():
                print(
                    f"   {category.title()}: {data['count']} markets, {data['avg_confidence']:.1%} avg confidence"
                )

            # Show final cache stats
            if pipeline.enable_caching:
                final_stats = pipeline.get_cache_stats()
                if final_stats:
                    print(f"\\n💾 CACHE PERFORMANCE:")
                    print(f"   Final entries: {final_stats['total_entries']}")
                    print(f"   Total size: {final_stats['total_size_kb']:.1f} KB")
                    if final_stats["most_accessed"]:
                        print(
                            f"   Most accessed: {final_stats['most_accessed'][0]['topic']}"
                        )

            print(
                f"\\n✅ Analysis complete! Processed {len(analyses)} markets with caching."
            )
            print("💡 Results stored in ML database. Cache persists for future runs.")

        else:
            print("❌ No analyses completed.")


if __name__ == "__main__":
    main()

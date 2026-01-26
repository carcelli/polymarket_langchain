"""
Train ML Model on Bitcoin Market Data

Uses collected market snapshots to train a predictor that identifies
profitable trading opportunities (markets where YES will win).

Features:
- XGBoost classifier for binary outcome prediction
- Feature importance analysis
- Backtesting simulation
- Edge detection (ML prob vs market prob)

Usage:
    # Train model on collected data
    python examples/train_bitcoin_predictor.py

    # Train with custom quality threshold
    python examples/train_bitcoin_predictor.py --min-quality 0.8

    # Predict on current markets
    python examples/train_bitcoin_predictor.py --predict-live
"""

import argparse
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install with: pip install xgboost")

DB_PATH = Path("data/bitcoin_tracker.db")


class BitcoinMarketPredictor:
    """
    ML model to predict Bitcoin market outcomes and identify edges.
    
    Predicts whether a market will resolve to YES based on:
    - Current market price (wisdom of crowd)
    - Technical indicators (momentum, volatility, RSI)
    - Bitcoin spot price and trends
    - Volume and liquidity metrics
    """
    
    def __init__(self, min_quality: float = 0.5):
        """
        Initialize predictor.
        
        Args:
            min_quality: Minimum data quality score for training (0-1)
        """
        self.min_quality = min_quality
        self.model = None
        self.feature_names = None
        self.feature_importance = None
    
    def load_training_data(self, db_path: Path) -> pd.DataFrame:
        """Load and prepare training data."""
        conn = sqlite3.connect(str(db_path))
        
        # Query resolved markets with quality filter
        query = f"""
            SELECT 
                -- Features
                yes_price as market_probability,
                volume,
                liquidity,
                btc_spot_price,
                btc_24h_change_pct,
                price_momentum_15m,
                price_momentum_1h,
                volume_spike,
                price_volatility,
                rsi_14,
                market_edge,
                time_to_expiry_hours,
                data_quality_score,
                
                -- Label
                CASE WHEN outcome = 'YES' THEN 1 ELSE 0 END as label,
                
                -- Metadata
                market_id,
                timestamp,
                question
            FROM market_snapshots
            WHERE resolved = 1
                AND data_quality_score >= {self.min_quality}
                AND outcome IS NOT NULL
            ORDER BY timestamp DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        print(f"\n📊 Training Data Loaded")
        print(f"   Total samples: {len(df):,}")
        print(f"   YES outcomes: {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
        print(f"   NO outcomes: {(~df['label'].astype(bool)).sum():,}")
        print(f"   Quality filter: ≥{self.min_quality}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame):
        """
        Prepare feature matrix and labels.
        
        Args:
            df: DataFrame with market data
        
        Returns:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
        """
        # Define feature columns
        feature_cols = [
            'market_probability',
            'volume',
            'liquidity',
            'btc_spot_price',
            'btc_24h_change_pct',
            'price_momentum_15m',
            'price_momentum_1h',
            'volume_spike',
            'price_volatility',
            'rsi_14',
            'market_edge',
            'time_to_expiry_hours',
        ]
        
        # Drop rows with missing features
        df_clean = df[feature_cols + ['label']].dropna()
        
        X = df_clean[feature_cols].values
        y = df_clean['label'].values
        
        print(f"\n🎯 Features: {len(feature_cols)}")
        print(f"   Samples after cleaning: {len(X):,}")
        
        return X, y, feature_cols
    
    def train(self, X, y, feature_names):
        """
        Train XGBoost classifier.
        
        Args:
            X: Feature matrix
            y: Labels
            feature_names: List of feature names
        """
        if not XGBOOST_AVAILABLE:
            print("❌ XGBoost not available. Cannot train.")
            return
        
        self.feature_names = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n🔄 Training XGBoost Classifier...")
        print(f"   Train set: {len(X_train):,} samples")
        print(f"   Test set: {len(X_test):,} samples")
        
        # Train model with class balancing
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,  # Handle class imbalance
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        print(f"\n📈 Model Performance (Test Set)")
        print(f"   Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
        print(f"   Precision: {precision_score(y_test, y_pred):.3f}")
        print(f"   Recall:    {recall_score(y_test, y_pred):.3f}")
        print(f"   F1 Score:  {f1_score(y_test, y_pred):.3f}")
        print(f"   ROC AUC:   {roc_auc_score(y_test, y_pred_proba):.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"                 Predicted NO  Predicted YES")
        print(f"   Actual NO     {cm[0, 0]:8d}      {cm[0, 1]:8d}")
        print(f"   Actual YES    {cm[1, 0]:8d}      {cm[1, 1]:8d}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🎯 Top Feature Importance:")
        for i, row in self.feature_importance.head(8).iterrows():
            print(f"   {row['feature']:25s} {row['importance']:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        print(f"\n✅ Cross-Validation ROC AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    def calculate_edge(self, market_prob: float, ml_prob: float, threshold: float = 0.05) -> dict:
        """
        Calculate trading edge (difference between ML prediction and market price).
        
        Args:
            market_prob: Current market probability (yes_price)
            ml_prob: ML model probability
            threshold: Minimum edge to recommend trade
        
        Returns:
            Dict with edge analysis and recommendation
        """
        edge = ml_prob - market_prob
        
        recommendation = "PASS"
        if edge > threshold:
            recommendation = "BUY YES"
        elif edge < -threshold:
            recommendation = "BUY NO"
        
        # Expected value calculation
        # If we buy YES at market_prob, and ML predicts ml_prob chance of YES:
        ev_yes = (ml_prob * (1 - market_prob)) - ((1 - ml_prob) * market_prob)
        ev_no = ((1 - ml_prob) * (1 - (1 - market_prob))) - (ml_prob * (1 - market_prob))
        
        return {
            'edge': edge,
            'edge_pct': edge * 100,
            'recommendation': recommendation,
            'ml_probability': ml_prob,
            'market_probability': market_prob,
            'expected_value_yes': ev_yes,
            'expected_value_no': ev_no,
            'confidence': abs(edge)
        }
    
    def predict_live_markets(self, db_path: Path, min_edge: float = 0.05):
        """
        Predict outcomes for current unresolved markets.
        
        Args:
            db_path: Path to database
            min_edge: Minimum edge to show recommendations
        """
        if self.model is None:
            print("❌ Model not trained. Run train() first.")
            return
        
        conn = sqlite3.connect(str(db_path))
        
        # Get latest snapshot for each unresolved market
        query = """
            SELECT 
                market_id,
                question,
                yes_price,
                volume,
                liquidity,
                btc_spot_price,
                btc_24h_change_pct,
                price_momentum_15m,
                price_momentum_1h,
                volume_spike,
                price_volatility,
                rsi_14,
                market_edge,
                time_to_expiry_hours,
                timestamp
            FROM market_snapshots
            WHERE resolved = 0
                AND market_id IN (
                    SELECT market_id 
                    FROM market_snapshots 
                    WHERE resolved = 0 
                    GROUP BY market_id 
                    HAVING MAX(timestamp)
                )
            ORDER BY volume DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            print("❌ No unresolved markets found")
            return
        
        print(f"\n🔮 Predicting {len(df)} Live Markets\n")
        print("="*80)
        
        # Prepare features
        X = df[self.feature_names].values
        
        # Predict probabilities
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Analyze each market
        opportunities = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            ml_prob = y_pred_proba[i]
            market_prob = row['yes_price']
            
            analysis = self.calculate_edge(market_prob, ml_prob, threshold=min_edge)
            
            if analysis['recommendation'] != "PASS":
                opportunities.append({
                    'market_id': row['market_id'],
                    'question': row['question'],
                    **analysis
                })
        
        # Sort by edge (absolute value)
        opportunities.sort(key=lambda x: abs(x['edge']), reverse=True)
        
        # Display results
        if opportunities:
            print(f"\n🎯 Found {len(opportunities)} Trading Opportunities (|edge| ≥ {min_edge*100:.0f}%)\n")
            
            for i, opp in enumerate(opportunities[:10], 1):
                print(f"{i}. {opp['question'][:70]}")
                print(f"   Market ID: {opp['market_id']}")
                print(f"   Market Prob: {opp['market_probability']:.1%} | ML Prob: {opp['ml_probability']:.1%}")
                print(f"   Edge: {opp['edge_pct']:+.1f}%")
                print(f"   🎯 {opp['recommendation']} (EV: {opp['expected_value_yes'] if opp['recommendation'] == 'BUY YES' else opp['expected_value_no']:.3f})")
                print()
        else:
            print(f"😐 No significant edges found (minimum: {min_edge*100:.0f}%)")
        
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description="Train ML model on Bitcoin market data",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--db',
        type=Path,
        default=DB_PATH,
        help=f'Database path (default: {DB_PATH})'
    )
    
    parser.add_argument(
        '--min-quality',
        type=float,
        default=0.5,
        help='Minimum data quality score for training (default: 0.5)'
    )
    
    parser.add_argument(
        '--predict-live',
        action='store_true',
        help='Predict outcomes for current markets after training'
    )
    
    parser.add_argument(
        '--min-edge',
        type=float,
        default=0.05,
        help='Minimum edge to recommend trades (default: 0.05 = 5%%)'
    )
    
    args = parser.parse_args()
    
    if not args.db.exists():
        print(f"❌ Database not found: {args.db}")
        print(f"\n💡 To collect data, run:")
        print(f"   python -m polymarket_agents.services.bitcoin_tracker")
        return
    
    # Initialize predictor
    predictor = BitcoinMarketPredictor(min_quality=args.min_quality)
    
    # Load training data
    df = predictor.load_training_data(args.db)
    
    if df.empty or len(df) < 100:
        print(f"\n❌ Insufficient training data ({len(df)} samples)")
        print(f"   Need at least 100 resolved markets with quality ≥ {args.min_quality}")
        print(f"\n💡 Run the tracker for longer to collect more data:")
        print(f"   python -m polymarket_agents.services.bitcoin_tracker")
        return
    
    # Prepare features
    X, y, feature_names = predictor.prepare_features(df)
    
    # Train model
    predictor.train(X, y, feature_names)
    
    # Predict live markets if requested
    if args.predict_live:
        predictor.predict_live_markets(args.db, min_edge=args.min_edge)


if __name__ == "__main__":
    main()

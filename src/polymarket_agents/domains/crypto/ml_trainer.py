"""
Crypto ML Trainer

Trains models on collected 15-minute market data.

Usage:
    python -m polymarket_agents.domains.crypto.ml_trainer
"""

import sqlite3
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib


DB_PATH = Path("data/crypto_ml.db")
MODEL_PATH = Path("data/models")


def load_training_data(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load resolved markets for training."""

    conn = sqlite3.connect(str(db_path))

    df = pd.read_sql_query(
        """
        SELECT
            asset,
            yes_price,
            no_price,
            volume,
            expiry_minutes,
            current_price,
            momentum_5m,
            momentum_30m,
            volatility,
            volume_spike,
            rsi,
            deviation,
            outcome
        FROM market_snapshots
        WHERE resolved = 1
        AND outcome IS NOT NULL
    """,
        conn,
    )

    conn.close()

    # Create target
    df["target"] = (df["outcome"] == "YES").astype(int)

    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare feature matrix and target."""

    # Feature columns
    feature_cols = [
        "yes_price",
        "no_price",
        "volume",
        "expiry_minutes",
        "momentum_5m",
        "momentum_30m",
        "volatility",
        "volume_spike",
        "rsi",
        "deviation",
    ]

    # Add asset one-hot encoding
    asset_dummies = pd.get_dummies(df["asset"], prefix="asset")
    df = pd.concat([df, asset_dummies], axis=1)

    # Add asset columns to features
    asset_cols = [c for c in df.columns if c.startswith("asset_")]
    all_features = feature_cols + asset_cols

    X = df[all_features].fillna(0)
    y = df["target"]

    return X, y, all_features


def train_model(X, y, model_type: str = "rf"):
    """Train a classifier."""

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if len(y.unique()) > 1 else None,
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    if model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "gb":
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        auc = 0.5  # Can't compute AUC with single class

    return model, scaler, accuracy, auc, X_test, y_test, y_pred


def main():
    """Train and evaluate models."""

    print("🔄 Loading training data...")
    df = load_training_data()

    print(f"   Total samples: {len(df)}")
    print("   Outcome distribution:")
    print(f"      YES: {(df['target'] == 1).sum()}")
    print(f"      NO:  {(df['target'] == 0).sum()}")

    if len(df) < 10:
        print("\n⚠️  Not enough data for training. Collect more samples.")
        print("   Run: python -m polymarket_agents.domains.crypto.data_collector")
        return

    # Prepare features
    X, y, feature_names = prepare_features(df)

    print(f"\n📊 Features: {len(feature_names)}")
    print(f"   {', '.join(feature_names[:5])}...")

    # Train Random Forest
    print("\n🌲 Training Random Forest...")
    rf_model, rf_scaler, rf_acc, rf_auc, X_test, y_test, y_pred = train_model(
        X, y, "rf"
    )

    print(f"   Accuracy: {rf_acc:.1%}")
    print(f"   AUC: {rf_auc:.3f}")

    # Feature importance
    print("\n📈 Top Features:")
    importance = pd.DataFrame(
        {"feature": feature_names, "importance": rf_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    for _, row in importance.head(5).iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")

    # Train Gradient Boosting
    print("\n🚀 Training Gradient Boosting...")
    gb_model, gb_scaler, gb_acc, gb_auc, _, _, _ = train_model(X, y, "gb")

    print(f"   Accuracy: {gb_acc:.1%}")
    print(f"   AUC: {gb_auc:.3f}")

    # Save best model
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    if rf_auc >= gb_auc:
        best_model, best_scaler, best_name = rf_model, rf_scaler, "RandomForest"
    else:
        best_model, best_scaler, best_name = gb_model, gb_scaler, "GradientBoosting"

    model_file = MODEL_PATH / "crypto_predictor.joblib"
    scaler_file = MODEL_PATH / "crypto_scaler.joblib"

    joblib.dump(best_model, model_file)
    joblib.dump(best_scaler, scaler_file)
    joblib.dump(feature_names, MODEL_PATH / "feature_names.joblib")

    print(f"\n✅ Saved {best_name} model to {model_file}")

    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["NO", "YES"]))


if __name__ == "__main__":
    main()

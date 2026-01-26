#!/usr/bin/env python3
"""
Predictive model for Up/Down markets.

Implements simple momentum + volume strategies with confidence scoring.
Start here and iterate - prove edge before going live.
"""

import ccxt
import numpy as np
from typing import Tuple, Optional, Literal
from datetime import datetime
import time


class SimplePredictor:
    """
    Baseline predictor using momentum + volume signals.

    Strategy:
    - Momentum: Recent price direction
    - Volume: Recent volume spikes
    - Confidence: Based on signal strength
    """

    def __init__(self, symbol: str = "BTC/USDT", exchange_id: str = "binance"):
        try:
            exchange_class = getattr(ccxt, exchange_id)
            self.exchange = exchange_class()
        except:
            print(f"⚠️  Warning: Could not initialize {exchange_id}, using mock data")
            self.exchange = None

        self.symbol = symbol

    def fetch_recent_data(self, timeframe: str = "1m", limit: int = 60):
        """Fetch recent OHLCV data."""
        if not self.exchange:
            # Mock data for testing
            return self._generate_mock_data(limit)

        try:
            bars = self.exchange.fetch_ohlcv(
                self.symbol, timeframe=timeframe, limit=limit
            )
            return np.array(bars, dtype=np.float32)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return self._generate_mock_data(limit)

    def _generate_mock_data(self, limit: int):
        """Generate mock OHLCV for testing."""
        # Simulate random walk price data
        base_price = 50000.0
        prices = base_price + np.cumsum(np.random.randn(limit) * 100)
        volumes = np.random.lognormal(10, 1, limit)

        data = np.column_stack(
            [
                np.arange(limit) * 60000,  # timestamps
                prices,  # open
                prices * 1.001,  # high
                prices * 0.999,  # low
                prices,  # close
                volumes,  # volume
            ]
        )
        return data

    def predict(
        self, asset: str = "BTC"
    ) -> Tuple[Optional[Literal["UP", "DOWN"]], float]:
        """
        Generate prediction for next 5-15 minute window.

        Returns:
            (direction, confidence) or (None, 0) if no strong signal
        """

        # Fetch data
        data = self.fetch_recent_data()

        if data is None or len(data) < 20:
            return None, 0.0

        closes = data[:, 4]
        volumes = data[:, 5]

        # Calculate features

        # 1. Short-term momentum (last 5 minutes)
        recent_return = (closes[-1] - closes[-5]) / closes[-5]

        # 2. Medium-term momentum (last 15 minutes)
        medium_return = (closes[-1] - closes[-15]) / closes[-15]

        # 3. Volume spike (compare to average)
        avg_volume = volumes[-30:-1].mean()
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # 4. Volatility (recent price swings)
        volatility = np.std(closes[-15:]) / closes[-15:].mean()

        # Simple scoring system
        score = 0.0

        # Momentum signals
        if recent_return > 0.001:  # 0.1% up
            score += 0.15
        elif recent_return < -0.001:
            score -= 0.15

        if medium_return > 0.002:  # 0.2% up
            score += 0.10
        elif medium_return < -0.002:
            score -= 0.10

        # Volume confirmation
        if volume_ratio > 1.5:  # Volume spike
            score += 0.10 * np.sign(recent_return)  # Amplify momentum direction

        # Reduce confidence in high volatility (less predictable)
        if volatility > 0.01:
            score *= 0.8

        # Convert score to direction and confidence
        base_confidence = 0.5
        confidence = base_confidence + abs(score)

        if score > 0.05:
            direction = "UP"
        elif score < -0.05:
            direction = "DOWN"
        else:
            # No strong signal
            return None, 0.0

        # Cap confidence at realistic levels
        confidence = min(confidence, 0.70)

        return direction, confidence

    def get_diagnostics(self) -> dict:
        """Get current market diagnostics for debugging."""
        data = self.fetch_recent_data()

        if data is None:
            return {}

        closes = data[:, 4]
        volumes = data[:, 5]

        return {
            "current_price": closes[-1],
            "price_change_5m": (closes[-1] - closes[-5]) / closes[-5],
            "price_change_15m": (closes[-1] - closes[-15]) / closes[-15],
            "volume_ratio": volumes[-1] / volumes[-30:-1].mean(),
            "volatility": np.std(closes[-15:]) / closes[-15:].mean(),
        }


class EnsemblePredictor:
    """
    Ensemble of multiple predictors for more robust signals.
    """

    def __init__(self):
        self.predictors = {
            "BTC": SimplePredictor("BTC/USDT"),
            "ETH": SimplePredictor("ETH/USDT"),
            "SOL": SimplePredictor("SOL/USDT"),
        }

    def predict(self, asset: str) -> Tuple[Optional[str], float]:
        """Get prediction for specific asset."""
        if asset.upper() in self.predictors:
            return self.predictors[asset.upper()].predict(asset)
        return None, 0.0

    def predict_all(self) -> dict:
        """Get predictions for all assets."""
        predictions = {}
        for asset, predictor in self.predictors.items():
            direction, confidence = predictor.predict(asset)
            predictions[asset] = {
                "direction": direction,
                "confidence": confidence,
                "diagnostics": predictor.get_diagnostics(),
            }
        return predictions


def monitor_with_predictions(check_interval: int = 30, min_confidence: float = 0.60):
    """
    Monitor markets and show predictions when they appear.
    Only alerts when prediction confidence exceeds threshold.
    """

    print("=" * 80)
    print("🤖 PREDICTIVE MARKET MONITOR")
    print("=" * 80)
    print(f"⏰ Checking every {check_interval} seconds")
    print(f"🎯 Min confidence: {min_confidence:.0%}")
    print("💡 Press Ctrl+C to stop\n")

    predictor = EnsemblePredictor()

    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")

            # Get predictions for all assets
            predictions = predictor.predict_all()

            # Show any strong signals
            strong_signals = []
            for asset, pred in predictions.items():
                if pred["direction"] and pred["confidence"] >= min_confidence:
                    strong_signals.append((asset, pred))

            if strong_signals:
                print(f"\n🚨 [{timestamp}] STRONG SIGNALS DETECTED!")
                for asset, pred in strong_signals:
                    print(f"\n   {asset}:")
                    print(f"      Direction: {pred['direction']}")
                    print(f"      Confidence: {pred['confidence']:.1%}")
                    diag = pred["diagnostics"]
                    if diag:
                        print(f"      Price: ${diag.get('current_price', 0):.2f}")
                        print(f"      5m change: {diag.get('price_change_5m', 0):+.2%}")
                        print(
                            f"      15m change: {diag.get('price_change_15m', 0):+.2%}"
                        )

                print(
                    f"\n   💡 If a market appears, consider betting {strong_signals[0][1]['direction']}"
                )
                print("      (Paper trade first to validate!)")

            time.sleep(check_interval)

    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        # Run continuous monitoring
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        min_conf = float(sys.argv[3]) if len(sys.argv) > 3 else 0.60
        monitor_with_predictions(interval, min_conf)

    else:
        # Single prediction
        predictor = EnsemblePredictor()
        predictions = predictor.predict_all()

        print("\n🔮 Current Predictions:\n")
        for asset, pred in predictions.items():
            print(f"{asset}:")
            if pred["direction"]:
                print(f"   Direction: {pred['direction']}")
                print(f"   Confidence: {pred['confidence']:.1%}")
            else:
                print("   No strong signal")

            diag = pred["diagnostics"]
            if diag:
                print(f"   Price: ${diag.get('current_price', 0):.2f}")
                print(f"   5m momentum: {diag.get('price_change_5m', 0):+.2%}")
            print()

        print("💡 Run with 'monitor' argument for continuous predictions:")
        print("   python scripts/predict_updown.py monitor [interval] [min_confidence]")

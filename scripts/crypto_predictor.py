#!/usr/bin/env python3
"""
Enhanced Crypto Up/Down Predictor

Production-grade predictor for 15-minute crypto markets.
Mirrors NBA predictor architecture for consistency.

Features:
- Multiple technical indicators
- Exchange orderbook analysis (future)
- Ensemble weighting
- Edge calculation vs market prices
"""

import ccxt
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, Dict


class CryptoPredictor:
    """
    Production crypto predictor for Up/Down markets.

    Similar to NBAPredictor but for ultra-short timeframes.
    """

    def __init__(self, exchange_id: str = "binance", lookback_minutes: int = 30):
        """
        Args:
            exchange_id: CCXT exchange (binance, coinbase, etc.)
            lookback_minutes: Historical window for indicators
        """
        self.exchange = getattr(ccxt, exchange_id)()
        self.lookback = lookback_minutes

        print(
            f"✅ Crypto Predictor initialized ({exchange_id}, {lookback_minutes}m window)"
        )

    def predict_direction(
        self,
        asset: str,
        market_up_price: float = 0.5,
        market_down_price: float = 0.5,
        duration_minutes: int = 15,
    ) -> Tuple[Optional[str], float, Dict]:
        """
        Predict direction with confidence and edge.

        Args:
            asset: 'BTC', 'ETH', 'SOL', etc.
            market_up_price: Market implied prob of UP
            market_down_price: Market implied prob of DOWN
            duration_minutes: Market duration (5, 15, 30, etc.)

        Returns:
            (direction, confidence, details) where:
            - direction: 'UP', 'DOWN', or None
            - confidence: 0-1 prediction confidence
            - details: Dict with price, indicators, edge, etc.
        """

        try:
            symbol = f"{asset}/USDT"

            # Fetch recent candles
            ohlcv = self.exchange.fetch_ohlcv(
                symbol, "1m", limit=self.lookback  # 1-minute candles
            )

            if len(ohlcv) < self.lookback:
                return None, 0.0, {"error": "Insufficient data"}

            # Extract prices
            closes = np.array([x[4] for x in ohlcv])
            volumes = np.array([x[5] for x in ohlcv])
            highs = np.array([x[2] for x in ohlcv])
            lows = np.array([x[3] for x in ohlcv])

            current_price = closes[-1]

            # Calculate indicators
            indicators = self._calculate_indicators(closes, volumes, highs, lows)

            # Score direction
            up_score, down_score = self._score_direction(indicators, duration_minutes)

            # Determine prediction
            if up_score > down_score and up_score > 0.5:
                direction = "UP"
                model_prob = up_score
                market_price = market_up_price
            elif down_score > up_score and down_score > 0.5:
                direction = "DOWN"
                model_prob = down_score
                market_price = market_down_price
            else:
                return None, 0.0, {"reason": "No clear signal", **indicators}

            # Calculate edge
            edge = model_prob - market_price

            # Confidence is based on signal strength
            confidence = abs(up_score - down_score)

            details = {
                "asset": asset,
                "current_price": current_price,
                "prediction": direction,
                "model_prob": model_prob,
                "market_price": market_price,
                "edge": edge,
                "confidence": confidence,
                "indicators": indicators,
                "timestamp": datetime.now().isoformat(),
            }

            return direction, confidence, details

        except Exception as e:
            return None, 0.0, {"error": str(e)}

    def _calculate_indicators(
        self,
        closes: np.ndarray,
        volumes: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
    ) -> Dict:
        """Calculate technical indicators."""

        # Momentum (5-min vs 30-min)
        momentum_5m = (closes[-5:].mean() - closes[-10:-5].mean()) / closes[
            -10:-5
        ].mean()
        momentum_30m = (closes[-1] - closes[0]) / closes[0]

        # Volatility
        volatility = closes.std() / closes.mean()

        # Volume trend (recent vs average)
        recent_volume = volumes[-5:].mean()
        avg_volume = volumes.mean()
        volume_spike = (recent_volume - avg_volume) / avg_volume

        # RSI (14-period)
        rsi = self._calculate_rsi(closes, period=14)

        # Mean reversion
        mean_price = closes.mean()
        deviation = (closes[-1] - mean_price) / mean_price

        # ATR (Average True Range) - normalized
        atr = self._calculate_atr(highs, lows, closes, period=14)
        atr_normalized = atr / closes[-1]

        return {
            "momentum_5m": momentum_5m,
            "momentum_30m": momentum_30m,
            "volatility": volatility,
            "volume_spike": volume_spike,
            "rsi": rsi,
            "deviation_from_mean": deviation,
            "atr_normalized": atr_normalized,
        }

    def _calculate_rsi(self, closes: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator."""

        if len(closes) < period + 1:
            return 50.0  # Neutral

        deltas = np.diff(closes)
        gains = deltas.copy()
        losses = deltas.copy()

        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)

        avg_gain = gains[-period:].mean()
        avg_loss = losses[-period:].mean()

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_atr(
        self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14
    ) -> float:
        """Calculate Average True Range."""

        if len(highs) < period + 1:
            return 0.0

        tr_list = []
        for i in range(1, len(highs)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            tr_list.append(tr)

        atr = np.mean(tr_list[-period:])
        return atr

    def _score_direction(self, indicators: Dict, duration: int) -> Tuple[float, float]:
        """
        Score UP vs DOWN probability.

        Returns:
            (up_score, down_score) each 0-1
        """

        up_score = 0.5  # Start neutral
        down_score = 0.5

        # Momentum signals (strongest for short timeframes)
        if indicators["momentum_5m"] > 0.001:  # 0.1% move
            up_score += 0.15
        elif indicators["momentum_5m"] < -0.001:
            down_score += 0.15

        # RSI (overbought/oversold)
        if indicators["rsi"] < 30:  # Oversold → reversal UP
            up_score += 0.1
        elif indicators["rsi"] > 70:  # Overbought → reversal DOWN
            down_score += 0.1

        # Mean reversion (if high volatility)
        if indicators["volatility"] > 0.01:  # 1% volatility
            if indicators["deviation_from_mean"] > 0.005:  # Above mean
                down_score += 0.1  # Expect reversion
            elif indicators["deviation_from_mean"] < -0.005:  # Below mean
                up_score += 0.1

        # Volume confirmation
        if indicators["volume_spike"] > 0.2:  # 20% above average
            # Volume confirms momentum
            if indicators["momentum_5m"] > 0:
                up_score += 0.05
            else:
                down_score += 0.05

        # Volatility adjustment (high volatility → less confident)
        if indicators["atr_normalized"] > 0.01:
            # Dampen extreme scores in volatile markets
            up_score = 0.5 + (up_score - 0.5) * 0.8
            down_score = 0.5 + (down_score - 0.5) * 0.8

        # Normalize to probabilities
        total = up_score + down_score
        up_score /= total
        down_score /= total

        return up_score, down_score


def test_predictor():
    """Test predictor on live markets."""

    predictor = CryptoPredictor()

    assets = ["BTC", "ETH", "SOL"]

    print("\n" + "=" * 80)
    print("🪙 CRYPTO PREDICTOR TEST")
    print("=" * 80 + "\n")

    for asset in assets:
        print(f"📊 {asset}:")

        # Simulate market prices (50/50 by default)
        direction, confidence, details = predictor.predict_direction(
            asset, market_up_price=0.50, market_down_price=0.50, duration_minutes=15
        )

        if direction:
            print(f"   Prediction: {direction}")
            print(f"   Model prob: {details['model_prob']:.1%}")
            print(f"   Confidence: {confidence:.1%}")
            print(f"   Edge: {details['edge']:+.1%}")
            print(f"   Current price: ${details['current_price']:,.2f}")

            # Show key indicators
            ind = details["indicators"]
            print(f"   Momentum 5m: {ind['momentum_5m']:+.2%}")
            print(f"   RSI: {ind['rsi']:.1f}")
            print(f"   Volume spike: {ind['volume_spike']:+.1%}")
        else:
            print("   No clear signal")
            if "error" in details:
                print(f"   Error: {details['error']}")

        print()


if __name__ == "__main__":
    test_predictor()

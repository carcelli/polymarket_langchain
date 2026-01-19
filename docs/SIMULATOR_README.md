# 🎯 Production Simulator - Paper Trading System

## Overview

**Production-grade paper trading simulator** that mirrors live trading flow without blockchain execution. Tracks live Up/Down markets, applies predictive strategies, places virtual bets, and computes P&L - all without capital exposure.

## Why This Matters

Polymarket Up/Down markets are **zero-sum with platform fees** = negative expectancy for random betting. This simulator lets you:

✅ **Prove edge before risking capital** (200+ markets minimum)  
✅ **Validate predictive strategies** in real market conditions  
✅ **Measure true performance** (win rate, Sharpe, drawdown)  
✅ **Iterate on signals** without financial risk  

## Architecture

```
Event-driven, thread-safe, persistent state:

┌─────────────────────────────────────────┐
│   Live Market Discovery (Gamma API)    │
│   ↓                                     │
│   Predictor Integration (momentum/ML)  │
│   ↓                                     │
│   Virtual Bet Placement (Kelly sizing) │
│   ↓                                     │
│   Outcome Polling (resolution)         │
│   ↓                                     │
│   P&L Calculation & Tracking           │
└─────────────────────────────────────────┘
         ↓
    SQLite Persistence
```

## Quick Start

### Run the Simulator
```bash
# Default: 60% confidence threshold, 2% Kelly risk, 30s poll
python scripts/monitor_simulator.py

# Custom: 65% confidence, 3% risk, 45s poll
python scripts/monitor_simulator.py 0.65 0.03 45
```

### What It Does
1. **Monitors** Gamma API for new Up/Down markets every 30s
2. **Predicts** direction using momentum/volume signals
3. **Bets** virtually when confidence > threshold
4. **Resolves** after market expiry (polls outcome)
5. **Tracks** P&L, win rate, profit factor in SQLite

### Stop and See Results
Press `Ctrl+C` to stop and see performance summary:
```
📊 SIMULATION SUMMARY
══════════════════════════════════════════════════════════════════════

💰 Portfolio:
   Starting bankroll: $1,000.00
   Current bankroll: $1,234.56
   Total P&L: +$234.56 (+23.5%)
   Peak bankroll: $1,289.00
   Current drawdown: 4.2%

📈 Performance:
   Total bets: 147
   Wins: 83
   Losses: 64
   Win rate: 56.5%

⚖️  Risk Metrics:
   Avg P&L per bet: +$1.60
   Avg win: +$8.20
   Avg loss: -$9.80
   Profit factor: 0.84

🎯 Assessment:
   ✅ Strategy shows promise - strong edge detected
   💡 Consider tiny live test ($10-20) to validate execution
```

## Database Schema

**Location**: `data/simulator.db`

### Tables

**trades** - All observed markets and virtual bets
```sql
- market_id: Unique market identifier
- asset: BTC, ETH, SOL, XRP
- duration_min: Market duration (5 or 15 minutes)
- start_time, end_time: Market window
- predicted_dir: 'UP' or 'DOWN' (NULL if no bet)
- confidence: Predictor confidence (0.0-1.0)
- virtual_bet_usd: Bet size (Kelly-sized)
- outcome_dir: Actual market direction
- actual_end_price: Final market price
- virtual_profit: +/- P&L for this bet
- resolved: Boolean flag
```

**portfolio** - Bankroll and aggregate stats
```sql
- virtual_bankroll: Current capital
- total_bets: Count of resolved bets
- winning_bets: Count of wins
- total_profit: Cumulative P&L
- max_bankroll: Peak value (for drawdown calc)
```

## Key Features

### 1. Automatic Market Discovery
Polls Gamma API every 30s for new Up/Down markets:
- Filters for 5-15 minute duration
- Extracts asset (BTC/ETH/SOL/XRP)
- Tracks start/end times

### 2. Predictor Integration
```python
from predict_updown import EnsemblePredictor

predictor = EnsemblePredictor()
direction, confidence = predictor.predict('BTC')

if direction and confidence >= min_confidence:
    place_virtual_bet(market, direction, confidence)
```

### 3. Kelly Criterion Position Sizing
```python
# Adapts based on observed win rate and profit factor
# Defaults to 2% fixed risk if < 20 bets
# Uses 1/4 Kelly for safety (caps at 5% max)

bet_size = calculate_kelly_bet()
```

### 4. Outcome Polling
- Waits for market expiry + 5 min grace period
- Polls Gamma API for resolution
- Parses winning outcome from token prices
- Calculates profit: 80% return on wins (after fees)

### 5. Thread-Safe Persistence
- All DB writes protected by threading.Lock
- Survives crashes (persistent state)
- Query anytime with sqlite3

## Success Metrics

**Target (200+ markets):**
- ✅ Win rate > 55%
- ✅ Total P&L > 0%
- ✅ Max drawdown < 20%
- ✅ Sharpe ratio > 0.5
- ✅ Profit factor > 1.2

**Minimum Acceptable:**
- Win rate > 52%
- Positive P&L
- Drawdown < 30%

**Red Flags:**
- Win rate < 50%
- Negative P&L
- Increasing drawdown over time
- Profit factor < 0.9

## Operational Notes

### Payout Modeling
Real Up/Down markets return ~80-90% on winners after platform fees:
- Buy at ~$0.50 per share
- Win = $1.00 payout = $0.50 profit per share
- Net: 80-90% return on capital

Simulator uses 80% conservative estimate.

### Latency Considerations
- Predictor must run < 30s after market open
- Use local inference only (no API calls in prediction path)
- Markets fill fast - early entry critical

### Market Schedule
Up/Down markets appear during US trading hours:
- **Peak**: 9 AM - 4 PM ET
- **Frequency**: Every 5-15 minutes (when demand exists)
- **Expect**: 50-200 markets/week

Run simulator during US hours for best data collection.

### Edge Cases

**No historical data**
- Pure forward simulation
- Can't backtest without live data
- Must run during active hours

**Resolution delays**
- Polymarket resolution can lag
- 5-minute grace period built in
- Retry logic for failures

**Outcome ambiguity**
- Parser checks token prices for winner
- Fallback to market resolution data
- Logs errors for manual review

## Query the Database

```bash
# Connect to database
sqlite3 data/simulator.db

# Total P&L
SELECT SUM(virtual_profit) FROM trades WHERE resolved=TRUE;

# Win rate
SELECT 
  SUM(CASE WHEN virtual_profit > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) 
FROM trades 
WHERE resolved=TRUE;

# Recent bets
SELECT 
  asset, predicted_dir, outcome_dir, virtual_profit, resolved_at
FROM trades 
WHERE resolved=TRUE 
ORDER BY resolved_at DESC 
LIMIT 20;

# Performance by asset
SELECT 
  asset,
  COUNT(*) as bets,
  SUM(CASE WHEN virtual_profit > 0 THEN 1 ELSE 0 END) as wins,
  SUM(virtual_profit) as total_pnl
FROM trades 
WHERE resolved=TRUE 
GROUP BY asset;
```

## Next Steps

### Phase 1: Data Collection (2-4 weeks)
```bash
# Run simulator daily during US market hours
python scripts/monitor_simulator.py

# Let it accumulate 200+ markets
# Check progress periodically with Ctrl+C
```

### Phase 2: Strategy Improvement
If win rate < 55% after 100+ bets:
1. **Add features**:
   - Order book imbalance
   - Funding rate extremes
   - Cross-exchange spreads
   - Volume-weighted momentum

2. **Train ML model**:
   - LSTM on 1-min OHLCV
   - Train on historical Binance data
   - Target: 53-55% accuracy

3. **Test offline**:
   - Export trades to CSV
   - Backtest new strategies
   - Measure improvement

### Phase 3: Live Validation (If edge proven)
```bash
# Only after ≥200 markets with positive metrics
# Start with $10-20 ONLY

# 1. Fund wallet
# 2. Run real executor (TBD)
# 3. Compare live vs simulator performance
```

## Integration Points

### Replace Mock Predictor
Edit `scripts/monitor_simulator.py`:
```python
def predict_direction(self, asset: str) -> Tuple[Optional[str], float]:
    """Get prediction for asset direction."""
    
    # Replace this with your real predictor
    # Examples:
    # - LSTM model inference
    # - Order book analysis
    # - Technical indicators
    # - ML ensemble
    
    from your_predictor import YourModel
    
    model = YourModel()
    return model.predict(asset)
```

### Add Custom Metrics
Database is open - add your own analytics:
```python
# Example: Consecutive wins/losses
with self.lock:
    self.conn.execute("""
        CREATE TABLE IF NOT EXISTS streaks (
            date TEXT,
            max_win_streak INTEGER,
            max_loss_streak INTEGER
        )
    """)
```

## Business Value

This simulator teaches you:
- ✅ **Real-time data pipelines** (market feeds)
- ✅ **Event-driven architecture** (async processing)
- ✅ **Time-series prediction** (direction forecasting)
- ✅ **Risk management** (position sizing, drawdown)
- ✅ **Performance measurement** (Sharpe, win rate)

**All directly transferable** to building forecasting agents for small businesses (inventory optimization, demand prediction, pricing strategies).

Your north star: **Sustainable value creation**, not speculation.

## Support

**Common Issues:**

Q: No markets appearing?  
A: Run during US market hours (9 AM - 4 PM ET)

Q: Markets not resolving?  
A: Some resolution delays are normal - wait 15 minutes

Q: How long to prove edge?  
A: Minimum 200 markets (2-4 weeks of daily running)

Q: When to go live?  
A: Only after 200+ markets with win rate > 55%, positive P&L, drawdown < 20%

## Files

- `scripts/monitor_simulator.py` - Main simulator (this system)
- `scripts/predict_updown.py` - Momentum predictor (integrate yours)
- `data/simulator.db` - SQLite database
- `docs/PAPER_TRADING_GUIDE.md` - Complete guide
- `docs/UPDOWN_MARKETS_GUIDE.md` - Market mechanics

---

**Remember**: This is research infrastructure. Run it, prove edge, iterate on predictions. Only consider live capital after mathematical edge is demonstrated.

🚀 Ship the simulator, accumulate data, prove expectancy.

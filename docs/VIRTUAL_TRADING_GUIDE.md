# 🎯 Complete Virtual Trading Ecosystem

## Overview

Three production-grade paper trading systems for different use cases:

| System | Best For | Markets | Features |
|--------|----------|---------|----------|
| **`virtual_trader.py`** | Multi-market unified system | NBA, crypto, politics, all | Planning agent, ML registry, auto strategy selection |
| **`nba_simulator.py`** | NBA-focused | NBA games only | Log5 predictor, home advantage, team records |
| **`monitor_simulator.py`** | Crypto Up/Down | 5-15 min crypto | Momentum predictor, ultra-short timeframes |

## 🚀 Quick Start

### Option 1: Unified Virtual Trader (Recommended)

**Best for**: Multi-market trading with strategy auto-selection

```bash
# Default: NBA + Crypto markets, 3% min edge
python scripts/virtual_trader.py

# NBA only with 5% edge threshold
python scripts/virtual_trader.py --markets nba --min-edge 0.05

# All markets with 2% edge
python scripts/virtual_trader.py --markets all --min-edge 0.02

# Custom settings
python scripts/virtual_trader.py \
  --markets nba crypto \
  --min-edge 0.03 \
  --bet-size 0.02 \
  --interval 120
```

**Features:**
- ✅ Auto-detects market type (NBA, crypto, politics)
- ✅ Selects best strategy per market
- ✅ Integrates planning agent + ML registry
- ✅ Kelly criterion position sizing
- ✅ Risk management (drawdown limits, circuit breakers)
- ✅ Performance tracking by strategy and market type

### Option 2: NBA Simulator

**Best for**: Pure NBA focus with sports-specific features

```bash
# Default: 5% edge, $50k min volume, $20 bets
python scripts/nba_simulator.py

# Conservative: 7% edge, $75k min volume
python scripts/nba_simulator.py 0.07 75000 15.0 300
```

**Features:**
- ✅ Log5 formula with home advantage
- ✅ Current NBA standings integrated
- ✅ Team record analysis
- ✅ Edge calculation vs market
- ✅ Extensible for injuries/rest/Elo

### Option 3: Crypto Up/Down Simulator

**Best for**: Ultra-short-term crypto markets

```bash
# Default: 60% confidence, 2% risk, 30s poll
python scripts/monitor_simulator.py

# Aggressive: 55% confidence, 3% risk
python scripts/monitor_simulator.py 0.55 0.03 30
```

**Features:**
- ✅ Momentum + volume signals
- ✅ 5-15 minute market windows
- ✅ Outcome polling and resolution
- ✅ Thread-safe persistence

## 📊 System Comparison

### Virtual Trader (Unified)

**Pros:**
- Multi-market support
- Strategy auto-selection
- Planning agent integration
- ML registry integration
- Comprehensive risk management

**Cons:**
- More complex
- Slower (polls every 60s)
- Requires more dependencies

**Best for:** Production-grade multi-market trading

### NBA Simulator

**Pros:**
- Sports-specific predictor
- Simple baseline (Log5)
- Proven inefficiencies
- Easy to enhance

**Cons:**
- NBA only
- Limited to game winners
- Needs enhancement for injuries

**Best for:** Proving edge in NBA before scaling

### Crypto Up/Down Simulator

**Pros:**
- Ultra-fast markets
- High volume opportunities
- Good for rapid iteration

**Cons:**
- Hard to predict (noise)
- Random walk dominates
- Lower signal-to-noise

**Best for:** Testing execution speed, not edge

## 💡 Recommended Strategy

### Phase 1: NBA Focus (Weeks 1-4)
```bash
# Run NBA simulator to prove baseline edge
python scripts/nba_simulator.py

# Goal: 20+ games, win rate > 52%, positive P&L
```

**Why NBA first:**
- Better signal (fundamentals matter)
- Longer horizons (less noise)
- Documented inefficiencies
- Extensible features

### Phase 2: Multi-Market (Weeks 5-8)
```bash
# Expand to unified trader once NBA edge proven
python scripts/virtual_trader.py --markets nba crypto
```

**Expand predictors:**
- Add injury data to NBA
- Add orderbook analysis to crypto
- Train ML models on historical data

### Phase 3: Production (After proving edge)
```bash
# Only after 200+ virtual bets with positive metrics
# Consider tiny live capital ($10-20)
```

## 📈 Performance Tracking

### Query Results

```bash
# Virtual Trader
sqlite3 data/virtual_trader.db "
  SELECT market_type, COUNT(*), SUM(profit), AVG(edge)
  FROM virtual_trades
  WHERE resolved=TRUE
  GROUP BY market_type
"

# NBA Simulator
sqlite3 data/nba_simulator.db "
  SELECT COUNT(*), 
         SUM(CASE WHEN virtual_profit > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
         SUM(virtual_profit)
  FROM nba_bets
  WHERE resolved=TRUE
"

# Crypto Simulator  
sqlite3 data/simulator.db "
  SELECT 
    COUNT(*) as bets,
    SUM(CASE WHEN virtual_profit > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
    SUM(virtual_profit) as total_pnl
  FROM trades
  WHERE resolved=TRUE
"
```

### Success Metrics

**Minimum (20+ bets):**
- Win rate > 52%
- Positive total P&L
- Max drawdown < 30%

**Target (50+ bets):**
- Win rate > 55%
- ROI > 5%
- Max drawdown < 20%
- Sharpe ratio > 0.5

**Strong Edge (100+ bets):**
- Win rate > 58%
- ROI > 10%
- Max drawdown < 15%
- Sharpe ratio > 1.0

## 🎯 Feature Comparison

| Feature | Virtual Trader | NBA Simulator | Crypto Simulator |
|---------|---------------|---------------|------------------|
| **Market Types** | All | NBA only | Crypto Up/Down |
| **Strategy Selection** | Auto | Fixed (Log5) | Fixed (momentum) |
| **Planning Agent** | ✅ Yes | ❌ No | ❌ No |
| **ML Registry** | ✅ Yes | ❌ No | ❌ No |
| **Kelly Sizing** | ✅ Adaptive | ✅ Fixed | ✅ Adaptive |
| **Risk Limits** | ✅ Full suite | ⚠️ Basic | ⚠️ Basic |
| **Multi-threading** | ❌ No | ❌ No | ✅ Yes |
| **Performance by Strategy** | ✅ Yes | ❌ No | ❌ No |
| **Extensibility** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |

## 🔧 Configuration

### Virtual Trader

```python
# scripts/virtual_trader.py

# Market filters
MARKET_TYPES = ['nba', 'crypto', 'politics']  # or ['all']

# Risk parameters
MIN_EDGE = 0.03  # 3% minimum edge
BET_SIZE_PCT = 0.02  # 2% of bankroll
MAX_POSITION_SIZE = 0.05  # 5% cap
MAX_OPEN_POSITIONS = 10
MAX_DAILY_LOSS_PCT = 0.10  # 10% daily stop
MAX_CONSECUTIVE_LOSSES = 5

# Polling
POLL_INTERVAL = 60  # seconds
```

### NBA Simulator

```python
# scripts/nba_simulator.py

# Thresholds
min_edge = 0.05  # 5% edge
min_volume = 50000  # $50k minimum
bet_amount = 20.0  # Fixed bet
poll_interval = 300  # 5 minutes
```

### Crypto Simulator

```python
# scripts/monitor_simulator.py

min_confidence = 0.60  # 60% confidence
risk_per_trade = 0.02  # 2% risk
poll_interval = 30  # 30 seconds
```

## 🎓 Enhancement Roadmap

### Immediate (Week 1)
- [ ] Add NBA injury data scraping
- [ ] Add rest day detection
- [ ] Improve outcome parsing

### Short-term (Weeks 2-4)
- [ ] Train LSTM on historical crypto data
- [ ] Add orderbook imbalance signals
- [ ] Implement Elo ratings for NBA

### Medium-term (Months 2-3)
- [ ] Planning agent full integration
- [ ] ML model auto-training pipeline
- [ ] Live market monitoring dashboard
- [ ] Slack/email alerts for high-edge opportunities

### Long-term (Months 3-6)
- [ ] Multi-leg parlay logic
- [ ] Live betting (in-game updates)
- [ ] Cross-market arbitrage detection
- [ ] Production live trading (if edge proven)

## 🚨 Important Notes

### Do NOT Fund Wallet Yet

Run virtual trading for **minimum 200 bets** before considering real capital.

**Required before live:**
- ✅ Win rate > 55%
- ✅ Positive P&L over 3+ months
- ✅ Max drawdown < 20%
- ✅ Documented edge explanation
- ✅ Risk management rules defined

### Which System to Use?

**Use Virtual Trader if:**
- You want multi-market exposure
- You have ML strategies ready
- You need comprehensive risk management

**Use NBA Simulator if:**
- You're starting fresh
- You want to prove edge in sports first
- You prefer simple, understandable baseline

**Use Crypto Simulator if:**
- You're testing execution speed
- You have momentum/ML signals ready
- You accept lower signal-to-noise

**Recommendation:** Start with NBA Simulator (easiest to prove edge), then graduate to Virtual Trader once confident.

## 📞 Support

**Common Issues:**

Q: No markets found?  
A: NBA Simulator needs active games. Crypto needs US market hours.

Q: Which is best for beginners?  
A: NBA Simulator - clearest signal, easiest to enhance.

Q: Can I run multiple simultaneously?  
A: Yes, but they use different databases - no conflicts.

Q: How long to prove edge?  
A: NBA: 20-30 games minimum. Crypto: 200+ markets minimum.

## 🎉 You're Set

You now have **three production-grade virtual trading systems** ready to run. Start with NBA Simulator, prove edge, then scale to unified Virtual Trader.

**Remember:** Your north star is building sustainable forecasting value for small businesses - not speculation. These are research tools to validate methodology.

🚀 **Ship it, prove edge, iterate.**

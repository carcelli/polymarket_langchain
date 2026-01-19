# 📊 Paper Trading & Backtesting Guide

## ⚠️ READ THIS FIRST

**DO NOT fund your wallet yet!** These ultra-short-term markets are essentially high-frequency gambling with negative expected value for most participants. You need to **prove edge in simulation first**.

## Why Paper Trade First?

1. **Platform takes a cut** - Built-in house edge
2. **Random walk dominates** - 5-15 min crypto moves are mostly noise
3. **Liquidity providers extract edge** - Market makers have advantages
4. **No demonstrated edge = guaranteed losses**

### Your North Star
Build sustainable, value-creating agents for small-business owners - not speculative trading. Treat this as a **controlled experiment** in time-series prediction and agent automation.

## 🎯 The Right Path Forward

### Phase 1: Paper Trading (Current)
**Goal**: Log 200+ virtual bets before considering real capital

**Tools Created:**
- ✅ `paper_trading_system.py` - Virtual betting with P&L tracking
- ✅ `backtest_updown.py` - Test strategies on historical data
- ✅ `predict_updown.py` - Simple momentum/volume predictor
- ✅ `auto_paper_trader.py` - Fully automated paper trading system

### Phase 2: Prove Edge (3-6 months)
**Success Criteria:**
- Win rate > 55% over 200+ bets
- Positive total P&L after 3 months
- Max drawdown < 20%
- Sharpe ratio > 0.5

### Phase 3: Live Testing (If Phase 2 succeeds)
**Start tiny:** $10-20 only
- Test real execution
- Measure slippage
- Verify resolution process
- Check gas costs

### Phase 4: Scale (If Phase 3 succeeds)
Only after consistent profitable real trading

## 🚀 Quick Start: Automated Paper Trading

### 1. Start the Automated System
```bash
# Default: $10 bets, 60% confidence threshold, check every 30s
python scripts/auto_paper_trader.py

# Custom settings: $20 bets, 65% confidence, check every 45s
python scripts/auto_paper_trader.py 20.0 0.65 45
```

This will:
- ✅ Monitor for new Up/Down markets
- ✅ Get predictions from momentum strategy
- ✅ Automatically place paper bets when confidence > threshold
- ✅ Track performance in SQLite database
- ✅ Show P&L summary when you stop (Ctrl+C)

### 2. Check Performance Anytime
```bash
python scripts/paper_trading_system.py summary
```

Output example:
```
📊 PAPER TRADING PERFORMANCE SUMMARY
══════════════════════════════════════════════════════════════════════

💰 P&L Metrics:
   Total P&L: +$23.45
   Average P&L per bet: +$0.47
   Best bet: +$9.50
   Worst bet: -$10.00

📈 Win Rate:
   Total bets: 50
   Wins: 28 (56.0%)
   Losses: 22

⚖️  Risk/Reward:
   Average win: +$8.75
   Average loss: -$9.25
   Profit factor: 0.95
```

## 📚 Manual Paper Trading

### Place a Virtual Bet
```bash
python scripts/paper_trading_system.py bet <market_id> <UP|DOWN> <amount> [confidence]

# Example: Bet $10 that Bitcoin goes UP with 65% confidence
python scripts/paper_trading_system.py bet 1234567 UP 10.0 0.65
```

### Resolve a Market
```bash
python scripts/paper_trading_system.py resolve <market_id> <ending_price> <starting_price>

# Example: Bitcoin ended at $93,124, started at $92,994
python scripts/paper_trading_system.py resolve 1234567 93124 92994
```

## 🔬 Backtesting

Test strategies on historical data:

```bash
python scripts/backtest_updown.py
```

This loads historical Up/Down markets from your database and simulates trading with different strategies.

### Create Custom Strategies

Edit `backtest_updown.py` and implement your strategy:

```python
def my_strategy(market) -> Tuple[str, float]:
    """
    Your strategy logic here.
    
    Returns:
        (direction, confidence) or (None, 0) if no signal
    """
    
    # Example: Bet UP if volume > threshold
    if market['volume'] > 10000:
        return 'UP', 0.60
    
    return None, 0.0

# Run backtest
bt = Backtest(starting_capital=1000.0)
markets = bt.load_historical_markets(limit=500)
results = bt.run_strategy(my_strategy, markets, bet_size=10.0)
```

## 🤖 Predictive Models

### Current Implementation: Simple Momentum

File: `scripts/predict_updown.py`

**Features:**
- 5-minute momentum
- 15-minute momentum
- Volume spike detection
- Volatility adjustment

**Usage:**
```bash
# Single prediction
python scripts/predict_updown.py

# Continuous monitoring with predictions
python scripts/predict_updown.py monitor [interval] [min_confidence]
```

### Improving the Predictor

**Key signals to add:**
1. **Order book imbalance** - Bid/ask depth skew
2. **Cross-exchange arbitrage** - Price differences
3. **Funding rate extremes** - Perpetual futures rates
4. **On-chain flows** - Whale transfers, exchange in/outflows
5. **Technical indicators** - RSI, MACD, Bollinger Bands

**Example: Adding order book data**
```python
def get_orderbook_signal(symbol='BTC/USDT'):
    """Calculate bid/ask imbalance."""
    exchange = ccxt.binance()
    orderbook = exchange.fetch_order_book(symbol)
    
    bid_volume = sum(bid[1] for bid in orderbook['bids'][:10])
    ask_volume = sum(ask[1] for ask in orderbook['asks'][:10])
    
    imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    
    # Imbalance > 0.2 suggests buying pressure
    if imbalance > 0.2:
        return 'UP', 0.65
    elif imbalance < -0.2:
        return 'DOWN', 0.65
    
    return None, 0.0
```

## 📊 Performance Metrics

### Key Metrics to Track

**Win Rate**
- Target: > 55%
- Minimum acceptable: > 52%
- Below 50% = losing strategy

**Profit Factor**
- Average win / Average loss
- Target: > 1.5
- Break-even: 1.0

**Sharpe Ratio**
- Return / Volatility
- Target: > 0.5
- Good: > 1.0
- Excellent: > 2.0

**Max Drawdown**
- Largest peak-to-trough decline
- Target: < 20%
- Acceptable: < 30%
- Danger zone: > 40%

**Kelly Criterion**
- Optimal bet sizing
- Formula: f = (p * b - q) / b
  - p = win rate
  - b = win/loss ratio
  - q = 1 - p

## ⚠️ Risk Management

### Position Sizing
```python
# Never risk more than 1-2% of capital per bet
capital = 1000.0
max_risk_per_bet = capital * 0.02  # 2% = $20

# Kelly criterion (fractional for safety)
win_rate = 0.56
avg_win = 9.0
avg_loss = 10.0
b = avg_win / avg_loss

kelly_fraction = (win_rate * b - (1 - win_rate)) / b
safe_kelly = kelly_fraction * 0.25  # Use 1/4 Kelly for safety

bet_size = capital * safe_kelly
```

### Stop-Loss Rules
1. **Daily loss limit**: Stop if down 5% in one day
2. **Consecutive losses**: Stop after 5 straight losses
3. **Drawdown limit**: Reduce bet size by 50% if down 15%

### Circuit Breakers
```python
# In auto_paper_trader.py, add:
if daily_loss > capital * 0.05:
    print("⚠️  Daily loss limit hit - stopping")
    sys.exit()

if consecutive_losses >= 5:
    print("⚠️  Too many consecutive losses - stopping")
    sys.exit()
```

## 📈 What Good Performance Looks Like

### Example: Profitable Strategy (200 bets)
```
Total P&L: +$245.60 (+24.6%)
Win Rate: 58.0% (116 wins, 84 losses)
Avg Win: +$11.20
Avg Loss: -$9.80
Profit Factor: 1.14
Max Drawdown: -12.5%
Sharpe Ratio: 0.73
```

### Example: Losing Strategy (200 bets)
```
Total P&L: -$157.30 (-15.7%)
Win Rate: 48.5% (97 wins, 103 losses)
Avg Win: +$9.50
Avg Loss: -$10.25
Profit Factor: 0.93
Max Drawdown: -28.3%
Sharpe Ratio: -0.42
```

## 🎓 Learning Resources

### Understand the Math
- Expected Value (EV) = (Win% × AvgWin) - (Loss% × AvgLoss)
- Edge = Your probability - Market price
- Kelly sizing = (Edge / Odds) × Bankroll

### Crypto-Specific Knowledge
- Order flow analysis
- Funding rate mechanics
- Cross-exchange arbitrage
- Whale watching (on-chain analytics)

### Tools to Explore
- **ccxt**: Multi-exchange API library
- **pandas-ta**: Technical analysis library
- **MLflow**: Experiment tracking
- **Optuna**: Hyperparameter optimization

## 🚨 Red Flags to Watch For

**Signs your strategy doesn't work:**
- Win rate trending down over time
- Increasing max drawdown
- Only profitable on small sample (<100 bets)
- Can't explain why it should work (no edge hypothesis)
- Results disappear when slippage/fees added

**Signs of overfitting:**
- 80%+ win rate in backtest
- Strategy has 10+ parameters
- Works on historical data, fails in paper trading
- Performance degrades immediately live

## ✅ Checklist Before Going Live

- [ ] 200+ paper trades logged
- [ ] Win rate > 55%
- [ ] Positive P&L over 3+ months
- [ ] Max drawdown < 20%
- [ ] Strategy has clear edge hypothesis
- [ ] Risk management rules defined
- [ ] Position sizing calculated
- [ ] Slippage/fees accounted for
- [ ] Emotional readiness confirmed
- [ ] Starting capital truly disposable

## 💡 Next Steps

1. **Run automated paper trading for 2-4 weeks**
   ```bash
   python scripts/auto_paper_trader.py
   ```

2. **Improve the predictor**
   - Add more features (orderbook, funding rates)
   - Train ML model on historical data
   - Test multiple strategies

3. **Backtest rigorously**
   - Minimum 1,000 simulated markets
   - Test on out-of-sample data
   - Calculate realistic Sharpe ratio

4. **Document everything**
   - Log all design decisions
   - Track what works and what doesn't
   - Maintain experiment journal

5. **Only then consider live trading**
   - Start with $10-20 maximum
   - Treat as final integration test
   - Be prepared to lose it all

## 📞 Questions?

This is a **research project**, not a get-rich-quick scheme. The goal is to learn agent automation, time-series prediction, and real-time data pipelines - skills directly transferable to your core mission of building forecasting agents for small businesses.

**Remember**: Paper trade until you prove edge. Your north star is building sustainable value, not gambling on crypto price movements.

Good luck! 🚀

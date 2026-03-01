# 🚀 Bitcoin Market Tracker - Quick Start Guide

## ✅ What You've Got

I've built you a **production-ready continuous data collection system** for Bitcoin (and any) prediction markets:

```
✅ bitcoin_tracker.py        - Main tracker daemon (900 lines)
✅ query_bitcoin_data.py     - Data query & export tool
✅ train_bitcoin_predictor.py - ML training example
✅ find_markets_to_track.py  - Helper to discover markets
✅ BITCOIN_TRACKER_README.md - Complete documentation
```

---

## 🎯 How It Works

```
Every 15 minutes:
  1. Fetch Bitcoin spot price (Binance)
  2. Fetch target markets (Polymarket)
  3. Calculate technical indicators (momentum, RSI, volatility)
  4. Save snapshot to SQLite database
  5. Track outcomes for ML training

After collecting data:
  → Train XGBoost model
  → Predict market outcomes
  → Find trading edges (ML prob vs market prob)
  → Recommend BUY YES, BUY NO, or PASS
```

---

## 🚀 **3-Step Workflow**

### **Step 1: Find Markets to Track**

First, discover what markets are currently active:

```bash
# Find top markets by volume
python scripts/python/find_markets_to_track.py

# Find Bitcoin-specific markets
python scripts/python/find_markets_to_track.py --keywords bitcoin btc

# Find any crypto markets
python scripts/python/find_markets_to_track.py --keywords crypto ethereum defi
```

**Output:**
```
✅ Found 15 Active Markets

1. Will Trump deport 250,000-500,000 people?
   ID: 12345
   Volume: $1,061,247
   Prices: Yes=0.881, No=0.119

2. Will Bitcoin reach $170,000 by December 31, 2025?
   ID: 574073
   Volume: $7,458,963
   Prices: Yes=0.0005, No=0.9995

🚀 To track these markets:
python -m polymarket_agents.services.bitcoin_tracker --market-ids 12345,574073
```

---

### **Step 2: Start Collecting Data**

Run the tracker continuously:

```bash
# Track specific markets (RECOMMENDED)
python -m polymarket_agents.services.bitcoin_tracker --market-ids 574073,12345,67890

# OR track all Bitcoin markets (auto-discover)
python -m polymarket_agents.services.bitcoin_tracker --keywords bitcoin btc

# OR track specific keywords
python -m polymarket_agents.services.bitcoin_tracker --keywords "over under" price
```

**Command-line options:**
```bash
--market-ids IDS    # Comma-separated market IDs (most reliable)
--interval 300      # Collection interval in seconds (default: 900 = 15 min)
--db PATH           # Custom database path
--once              # Collect one snapshot and exit (for testing)
--verbose           # Debug logging
```

**What you'll see:**
```
✅ BitcoinMarketTracker initialized
   Database: data/bitcoin_tracker.db
   Interval: 900s (15.0 min)

🚀 Starting Bitcoin market tracker
📸 Collecting snapshot...
₿  BTC: $87,847.07 (-1.29% 24h)
📊 Found 8 markets
   📈 Will Bitcoin reach $170,000... (+0.023)
✅ Collected 8/8 markets
💾 Database: 8 total snapshots
😴 Sleeping 900s until next collection...
```

**Pro tip:** Run in background:
```bash
# With nohup
nohup python -m polymarket_agents.services.bitcoin_tracker --market-ids YOUR_IDS > tracker.log 2>&1 &

# Check it's running
tail -f tracker.log
```

---

### **Step 3: Train ML Model & Find Edges**

After collecting data for a few days/weeks:

```bash
# View your collected data
python scripts/python/query_bitcoin_data.py --stats

# Train model on resolved markets
python examples/train_bitcoin_predictor.py

# Get live predictions
python examples/train_bitcoin_predictor.py --predict-live --min-edge 0.05
```

**Example output:**
```
📊 Training Data Loaded
   Total samples: 1,234
   YES outcomes: 456 (37.0%)

📈 Model Performance (Test Set)
   Accuracy:  0.723
   ROC AUC:   0.781

🎯 Top Feature Importance:
   market_probability        0.3245
   btc_24h_change_pct       0.1834
   volume                    0.1423

🔮 Predicting 8 Live Markets

🎯 Found 3 Trading Opportunities (edge ≥ 5%)

1. Will Bitcoin reach $150,000 by March 31, 2026?
   Market Prob: 12.0% | ML Prob: 28.5%
   Edge: +16.5%
   🎯 BUY YES (Expected Value: 0.163)
```

---

## 📊 **Database Schema (ML-Ready)**

Your SQLite database stores:

```sql
market_snapshots (main table):
  - timestamp, market_id, question
  - yes_price, no_price, volume, liquidity
  - btc_spot_price, btc_24h_change_pct
  - price_momentum_15m, price_momentum_1h
  - volume_spike, price_volatility, rsi_14
  - resolved, outcome (for ML labels)
  - data_quality_score (0-1)
```

**Query examples:**
```python
import pandas as pd
import sqlite3

conn = sqlite3.connect('data/bitcoin_tracker.db')

# Load all data
df = pd.read_sql_query("SELECT * FROM market_snapshots", conn)

# Time series for specific market
df_series = pd.read_sql_query("""
    SELECT timestamp, yes_price, volume, btc_spot_price
    FROM market_snapshots
    WHERE market_id = '574073'
    ORDER BY timestamp
""", conn)

# ML-ready dataset (resolved markets only)
df_ml = pd.read_sql_query("""
    SELECT * FROM market_snapshots
    WHERE resolved = 1 AND data_quality_score >= 0.8
""", conn)
```

---

## 🔧 **Common Use Cases**

### **A. Track Bitcoin "Over/Under $X by Date" Markets**

```bash
# 1. Find markets
python scripts/python/find_markets_to_track.py --keywords "bitcoin" "reach" "above"

# 2. Copy market IDs from output, then track
python -m polymarket_agents.services.bitcoin_tracker --market-ids <IDS_HERE>
```

---

### **B. Track Any Crypto Markets**

```bash
# Track all crypto (Bitcoin, Ethereum, etc.)
python scripts/python/find_markets_to_track.py --keywords crypto btc eth sol
```

---

### **C. Export Data for External Analysis**

```bash
# Export to CSV
python scripts/python/query_bitcoin_data.py --export csv --output my_data.csv

# Export ML-ready dataset (high quality only)
python scripts/python/query_bitcoin_data.py --ml-ready --min-quality 0.8 --output ml_dataset.csv

# Then use in Excel, R, or other tools
```

---

### **D. Backtest Trading Strategy**

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect('data/bitcoin_tracker.db')

# Get resolved markets
df = pd.read_sql_query("""
    SELECT * FROM market_snapshots
    WHERE resolved = 1
""", conn)

# Simple strategy: Buy YES if price < 0.3
df['bought_yes'] = df['yes_price'] < 0.3
df['profit'] = df.apply(lambda row:
    (1 - row['yes_price']) if (row['bought_yes'] and row['outcome'] == 'YES')
    else -row['yes_price'] if row['bought_yes']
    else 0,
    axis=1
)

print(f"Total profit: ${df['profit'].sum():,.2f}")
print(f"Win rate: {(df[df['bought_yes']]['outcome'] == 'YES').mean():.1%}")
```

---

## 🚨 **Troubleshooting**

### **"No markets found"**

The Polymarket API might not have active markets matching your criteria. Try:
1. Broaden search: remove `--keywords` filter
2. Lower volume threshold: `--min-volume 100`
3. Check specific market IDs manually on polymarket.com

---

### **"Skipping market: no price data"**

This means the market is closed/resolved. The tracker automatically skips these. This is normal for old markets.

---

### **"Insufficient training data"**

You need at least ~100 resolved markets with quality ≥ 0.5 to train a reliable model. Keep the tracker running for several days/weeks.

---

### **"Database locked"**

Another process is accessing the database. Stop other processes or increase the timeout in the code.

---

## 📈 **What Makes This Production-Ready**

✅ **Robust Error Handling**
- Network failures → Retries with backoff
- API rate limits → Respects limits
- Missing data → Logs and continues

✅ **Data Quality**
- Quality scores (0-1) for each snapshot
- Filters low-liquidity markets
- Tracks data completeness

✅ **Graceful Shutdown**
- Ctrl+C → Saves data, closes connections
- SIGTERM → Same (for systemd/Docker)

✅ **Monitoring**
- Structured logging
- Collection run tracking
- Database statistics

✅ **Scalability**
- SQLite with indexes
- Efficient queries
- Configurable intervals

---

## 🎓 **ML Features Explained**

| Feature | Description | Use in ML |
|---------|-------------|-----------|
| `market_probability` | Current yes_price (0-1) | Baseline crowd wisdom |
| `btc_spot_price` | Real-time BTC price | Context for BTC markets |
| `btc_24h_change_pct` | 24h BTC price change | Market sentiment |
| `price_momentum_15m` | Price change last 15 min | Short-term trend |
| `price_momentum_1h` | Price change last hour | Medium-term trend |
| `volume_spike` | Volume vs average | Unusual activity detection |
| `price_volatility` | Std dev of prices | Uncertainty measure |
| `rsi_14` | Relative Strength Index | Overbought/oversold |
| `market_edge` | abs(price - 0.5) | Confidence of crowd |
| `time_to_expiry_hours` | Hours until close | Urgency factor |

---

## 📚 **Next Steps**

1. **Week 1**: Run tracker continuously, collect 500+ snapshots
2. **Week 2**: Monitor data quality, adjust interval if needed
3. **Week 3**: Train initial model, evaluate performance
4. **Week 4**: Tune hyperparameters, test live predictions
5. **Month 2+**: Deploy automated trading (paper first!)

---

## 🤝 **Integration with Your Existing System**

This tracker is designed to work with your codebase:

```python
# Use with your LangChain agents
from polymarket_agents.langchain.agent import create_ml_forecast_comparison_agent

agent = create_ml_forecast_comparison_agent()
result = agent.invoke({
    "messages": [HumanMessage(content="Analyze Bitcoin markets using tracker data")]
})

# Use with your AutoML tools
from polymarket_agents.automl.ml_tools import ModelTrainingTool

trainer = ModelTrainingTool()
trainer.run(
    model_type="MarketPredictor",
    experiment_name="bitcoin_prediction"
)

# Use with your memory manager
from polymarket_agents.memory.manager import MemoryManager

memory = MemoryManager()
markets = memory.list_top_volume_markets(category='crypto')
```

---

## 📖 **Full Documentation**

- **BITCOIN_TRACKER_README.md** - Complete technical reference (100+ lines)
- **Source code comments** - Detailed inline documentation

---

## 🎉 **You're Ready!**

You now have a professional-grade data collection pipeline. Start with:

```bash
# 1. Find markets
python scripts/python/find_markets_to_track.py

# 2. Start tracking (replace IDS with actual market IDs)
python -m polymarket_agents.services.bitcoin_tracker --market-ids IDS

# 3. Check progress after a few hours
python scripts/python/query_bitcoin_data.py --stats

# 4. Train model after a few days
python examples/train_bitcoin_predictor.py --predict-live
```

**Questions? Check:**
1. BITCOIN_TRACKER_README.md (detailed docs)
2. Source code comments (inline explanations)
3. Example scripts (working code)

Happy trading! 🚀📈

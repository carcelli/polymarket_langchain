# 🪙 Bitcoin Market Continuous Tracker

**Production-ready system for collecting, storing, and training ML models on Bitcoin prediction market data.**

---

## 🎯 **What It Does**

Continuously polls Polymarket for Bitcoin price prediction markets and stores historical snapshots with:
- ✅ **Market prices** (Yes/No probabilities)
- ✅ **Bitcoin spot price** (real-time from Binance)
- ✅ **Technical indicators** (momentum, RSI, volatility)
- ✅ **Volume metrics** (spikes, liquidity)
- ✅ **Outcome tracking** (for supervised ML training)

**Use cases:**
1. **ML Training**: Build predictive models on historical data
2. **Backtesting**: Simulate trading strategies
3. **Live Trading**: Identify mispriced markets in real-time
4. **Research**: Analyze prediction market efficiency

---

## 📁 **Files Created**

```
src/polymarket_agents/services/
  └── bitcoin_tracker.py          # Main tracker daemon (900 lines)

scripts/python/
  └── query_bitcoin_data.py       # Query & export utility

examples/
  └── train_bitcoin_predictor.py  # ML training example

data/
  └── bitcoin_tracker.db          # SQLite database (created on first run)
```

---

## 🚀 **Quick Start**

### **1. Start Collecting Data (15-min intervals)**

```bash
# Run continuously (Ctrl+C to stop)
python -m polymarket_agents.services.bitcoin_tracker

# Custom interval (5 minutes)
python -m polymarket_agents.services.bitcoin_tracker --interval 300

# Collect one snapshot and exit
python -m polymarket_agents.services.bitcoin_tracker --once
```

**What you'll see:**
```
✅ BitcoinMarketTracker initialized
   Database: data/bitcoin_tracker.db
   Interval: 900s (15.0 min)
🚀 Starting Bitcoin market tracker
   Press Ctrl+C to stop gracefully

📸 Collecting snapshot...
₿  BTC: $102,345.67 (+2.34% 24h)
📊 Found 8 Bitcoin markets
   📈 Will Bitcoin reach $170,000 by Dec 31, 2025? (+0.023)
✅ Collected 8/8 markets
💾 Database: 8 total snapshots
😴 Sleeping 900s until next collection...
```

---

### **2. Track Specific Markets**

If you know the market IDs you want to track:

```bash
python -m polymarket_agents.services.bitcoin_tracker --market-ids 574073,12345,67890
```

---

### **3. Check Your Data**

```bash
# Show statistics
python scripts/python/query_bitcoin_data.py --stats

# View specific market history
python scripts/python/query_bitcoin_data.py --market 574073

# Export to CSV for analysis
python scripts/python/query_bitcoin_data.py --export csv --output btc_data.csv

# Get ML-ready dataset (high quality only)
python scripts/python/query_bitcoin_data.py --ml-ready --min-quality 0.8
```

**Example output:**
```
======================================================================
📊 BITCOIN MARKET TRACKER - DATABASE STATISTICS
======================================================================

📸 Total Snapshots: 1,234
🎯 Unique Markets: 12
📅 Date Range: 2026-01-26 to 2026-02-10

✅ Resolved Markets: 234
⏳ Unresolved Markets: 1,000

🔥 Most Tracked Markets:
  1. Will Bitcoin reach $170,000 by December 31, 2025?
     Snapshots: 156, Price Range: 0.002-0.054

📈 Data Quality Distribution:
  High (≥0.8): 894 snapshots
  Medium (0.5-0.8): 298 snapshots
  Low (<0.5): 42 snapshots

⏰ Last 24 Hours: 192 snapshots
```

---

### **4. Train ML Model**

After collecting data for a few days/weeks:

```bash
# Train XGBoost classifier
python examples/train_bitcoin_predictor.py

# Predict live market opportunities
python examples/train_bitcoin_predictor.py --predict-live --min-edge 0.05
```

**Example output:**
```
📊 Training Data Loaded
   Total samples: 1,234
   YES outcomes: 456 (37.0%)
   NO outcomes: 778

🔄 Training XGBoost Classifier...
   Train set: 987 samples
   Test set: 247 samples

📈 Model Performance (Test Set)
   Accuracy:  0.723
   Precision: 0.689
   Recall:    0.654
   F1 Score:  0.671
   ROC AUC:   0.781

🎯 Top Feature Importance:
   market_probability        0.3245
   btc_24h_change_pct       0.1834
   volume                    0.1423
   price_momentum_1h         0.0982

🔮 Predicting 8 Live Markets

🎯 Found 3 Trading Opportunities (|edge| ≥ 5%)

1. Will Bitcoin reach $150,000 by March 31, 2026?
   Market Prob: 12.0% | ML Prob: 28.5%
   Edge: +16.5%
   🎯 BUY YES (EV: 0.163)
```

---

## 📊 **Database Schema**

### **market_snapshots** (Main table)
```sql
timestamp              TEXT    -- ISO 8601 timestamp
market_id              TEXT    -- Polymarket market ID
question               TEXT    -- Market question
yes_price              REAL    -- Current Yes price (0-1)
no_price               REAL    -- Current No price
implied_probability    REAL    -- yes_price as probability
volume                 REAL    -- Total trading volume
liquidity              REAL    -- Current liquidity
btc_spot_price         REAL    -- Bitcoin spot price (USDT)
btc_24h_change_pct     REAL    -- 24h Bitcoin price change

-- Technical Indicators (ML features)
price_momentum_15m     REAL    -- Price change over 15 min
price_momentum_1h      REAL    -- Price change over 1 hour
volume_spike           REAL    -- Volume vs average
price_volatility       REAL    -- Standard deviation
rsi_14                 REAL    -- Relative Strength Index
market_edge            REAL    -- Deviation from 0.5

-- Outcome Tracking (ML labels)
resolved               INTEGER -- 0=active, 1=resolved
outcome                TEXT    -- 'YES' or 'NO'
profit_if_bought_yes   REAL    -- Retrospective profit
profit_if_bought_no    REAL    

-- Metadata
data_quality_score     REAL    -- 0-1 quality metric
time_to_expiry_hours   REAL    -- Hours until market closes
```

### **market_resolutions** (Outcome tracking)
```sql
market_id         TEXT    PRIMARY KEY
resolved_at       TEXT    Timestamp of resolution
outcome           TEXT    'YES' or 'NO'
final_btc_price   REAL    Bitcoin price at resolution
```

### **collection_runs** (Monitoring)
```sql
start_time             TEXT
end_time               TEXT
snapshots_collected    INTEGER
errors                 INTEGER
status                 TEXT -- 'running', 'completed', 'failed'
```

---

## 🎓 **How It Works**

### **1. Data Collection Loop**

```python
while True:
    # 1. Fetch Bitcoin spot price (Binance)
    btc_price = fetch_binance_btc_price()
    
    # 2. Fetch active Bitcoin markets (Polymarket)
    markets = fetch_polymarket_markets(keywords=['bitcoin', 'btc'])
    
    # 3. For each market:
    for market in markets:
        # Calculate technical indicators
        indicators = calculate_indicators(market, history)
        
        # Save snapshot to database
        save_to_db(market, btc_price, indicators)
    
    # 4. Check for resolved markets
    check_resolutions()
    
    # 5. Sleep until next interval (default: 15 min)
    sleep(900)
```

---

### **2. Technical Indicators**

**Price Momentum** (15min, 1hour)
- Change in `yes_price` over time windows
- Detects rapid market movements

**Volume Spike**
- `(current_volume - avg_volume) / avg_volume`
- Identifies unusual trading activity

**Price Volatility**
- Standard deviation of recent prices
- Measures market uncertainty

**RSI (Relative Strength Index)**
- Oscillator between 0-100
- >70 = overbought, <30 = oversold

**Market Edge**
- `abs(yes_price - 0.5)`
- Deviation from "fair" 50/50 probability

---

### **3. ML Features → Prediction**

```
Input Features:          ML Model          Output:
  market_probability  →                 → probability_yes (0-1)
  volume              →   XGBoost       → edge (ML - market)
  btc_spot_price      →  Classifier     → recommendation
  momentum            →                 →   - BUY YES
  rsi_14              →                 →   - BUY NO
  ...                 →                 →   - PASS
```

**Training Pipeline:**
1. Filter data by quality score
2. Use resolved markets as labels
3. Train XGBoost classifier
4. Backtest on hold-out set
5. Deploy for live predictions

---

## ⚙️ **Configuration**

### **Environment Variables**

```bash
# Optional: If using specific Polymarket API key
export POLYMARKET_API_KEY="your-key"

# Optional: Custom database location
export BITCOIN_TRACKER_DB="data/my_custom.db"
```

---

### **Command-Line Options**

```bash
python -m polymarket_agents.services.bitcoin_tracker --help
```

| Option | Description | Default |
|--------|-------------|---------|
| `--interval SECONDS` | Collection interval | 900 (15 min) |
| `--market-ids IDS` | Track specific markets (comma-separated) | All Bitcoin markets |
| `--db PATH` | Database file path | `data/bitcoin_tracker.db` |
| `--once` | Collect one snapshot and exit | Continuous loop |
| `--verbose` | Enable debug logging | INFO level |

---

## 🔧 **Advanced Usage**

### **Run as Background Daemon (Linux/Mac)**

```bash
# With nohup
nohup python -m polymarket_agents.services.bitcoin_tracker > tracker.log 2>&1 &

# With systemd (create /etc/systemd/system/bitcoin-tracker.service)
[Unit]
Description=Bitcoin Market Tracker
After=network.target

[Service]
Type=simple
User=yourusername
WorkingDirectory=/path/to/polymarket_langchain
ExecStart=/usr/bin/python3 -m polymarket_agents.services.bitcoin_tracker
Restart=always

[Install]
WantedBy=multi-user.target

# Then:
sudo systemctl enable bitcoin-tracker
sudo systemctl start bitcoin-tracker
sudo systemctl status bitcoin-tracker
```

---

### **Scheduled Collection (Alternative to Continuous)**

If you prefer cron-based collection instead of a daemon:

```bash
# Run every 15 minutes
*/15 * * * * cd /path/to/polymarket_langchain && python -m polymarket_agents.services.bitcoin_tracker --once >> /var/log/bitcoin_tracker.log 2>&1
```

---

### **Export Data to Pandas**

```python
import pandas as pd
import sqlite3

conn = sqlite3.connect('data/bitcoin_tracker.db')

# Load all data
df = pd.read_sql_query("SELECT * FROM market_snapshots", conn)

# Filter high-quality resolved markets
df_ml = pd.read_sql_query("""
    SELECT * FROM market_snapshots
    WHERE resolved = 1
    AND data_quality_score >= 0.8
""", conn)

# Time series for specific market
df_series = pd.read_sql_query("""
    SELECT timestamp, yes_price, volume, btc_spot_price
    FROM market_snapshots
    WHERE market_id = '574073'
    ORDER BY timestamp
""", conn)

# Plot price evolution
df_series['timestamp'] = pd.to_datetime(df_series['timestamp'])
df_series.set_index('timestamp').plot(y='yes_price', title='Market Price Evolution')
```

---

## 🧪 **Testing & Validation**

### **1. Test Single Snapshot**
```bash
python -m polymarket_agents.services.bitcoin_tracker --once --verbose
```

### **2. Verify Data Quality**
```python
import sqlite3
conn = sqlite3.connect('data/bitcoin_tracker.db')
cursor = conn.cursor()

# Check for nulls in critical features
cursor.execute("""
    SELECT 
        COUNT(*) as total,
        SUM(CASE WHEN yes_price IS NULL THEN 1 ELSE 0 END) as null_price,
        SUM(CASE WHEN btc_spot_price IS NULL THEN 1 ELSE 0 END) as null_btc
    FROM market_snapshots
""")
print(cursor.fetchone())
```

### **3. Validate Indicators**
```bash
# Check RSI values are in [0, 100]
SELECT MIN(rsi_14), MAX(rsi_14) FROM market_snapshots WHERE rsi_14 IS NOT NULL;

# Check for outliers
SELECT * FROM market_snapshots WHERE price_volatility > 0.5 OR volume_spike > 10;
```

---

## 🚨 **Error Handling**

The tracker handles errors gracefully:

- **Network failures**: Retries with exponential backoff
- **API rate limits**: Respects rate limits, logs warnings
- **Missing data**: Logs skipped markets, continues collection
- **Database locks**: Uses SQLite timeout, retries on lock
- **Keyboard interrupt (Ctrl+C)**: Saves final data, closes connections

**Logging:**
```
[2026-01-26 14:30:15] INFO: 📸 Collecting snapshot...
[2026-01-26 14:30:16] INFO: ₿  BTC: $102,345.67 (+2.34% 24h)
[2026-01-26 14:30:17] WARNING: ⚠️  Skipping market 123: no price data
[2026-01-26 14:30:18] INFO: ✅ Collected 7/8 markets
[2026-01-26 14:30:19] INFO: 😴 Sleeping 900s until next collection...
```

---

## 📈 **Production Best Practices**

### **1. Data Quality Monitoring**
```sql
-- Check collection health
SELECT 
    DATE(timestamp) as date,
    COUNT(*) as snapshots_per_day,
    AVG(data_quality_score) as avg_quality
FROM market_snapshots
GROUP BY DATE(timestamp)
ORDER BY date DESC;
```

### **2. Disk Space Management**
```bash
# Database size
ls -lh data/bitcoin_tracker.db

# Archive old data (keep last 90 days)
sqlite3 data/bitcoin_tracker.db "DELETE FROM market_snapshots WHERE datetime(timestamp) < datetime('now', '-90 days')"
```

### **3. Alerting**
```python
# Example: Alert on stale data
import sqlite3
from datetime import datetime, timedelta

conn = sqlite3.connect('data/bitcoin_tracker.db')
cursor = conn.cursor()

cursor.execute("""
    SELECT MAX(timestamp) as last_snapshot
    FROM market_snapshots
""")
last = cursor.fetchone()[0]
last_dt = datetime.fromisoformat(last)

if datetime.now() - last_dt > timedelta(hours=1):
    send_alert("Bitcoin tracker: No data for 1+ hour!")
```

---

## 🎯 **Roadmap**

Future enhancements:
- [ ] Multi-exchange price aggregation (Coinbase, Kraken)
- [ ] Sentiment analysis from crypto news
- [ ] On-chain data integration (Polygon blockchain)
- [ ] Real-time web dashboard (Streamlit/Gradio)
- [ ] Automatic outcome resolution via smart contracts
- [ ] Multi-asset support (ETH, SOL, etc.)
- [ ] Time-series forecasting (LSTM, Prophet)

---

## 🤝 **Integration with Existing System**

This tracker complements your existing codebase:

```python
# Use with your ML tools
from polymarket_agents.automl.ml_tools import ModelTrainingTool
from polymarket_agents.services.bitcoin_tracker import BitcoinMarketTracker

# Use with your agents
from polymarket_agents.langchain.agent import create_ml_forecast_comparison_agent

agent = create_ml_forecast_comparison_agent()
result = agent.invoke({
    "messages": [HumanMessage(content="Analyze Bitcoin markets using tracker data")]
})

# Use with your memory manager
from polymarket_agents.memory.manager import MemoryManager

memory = MemoryManager()
markets = memory.list_top_volume_markets(category='crypto')
```

---

## 📚 **References**

- **Polymarket Gamma API**: https://gamma-api.polymarket.com/docs
- **Binance API (ccxt)**: https://docs.ccxt.com/
- **XGBoost**: https://xgboost.readthedocs.io/
- **Prediction Market Theory**: Hanson, R. (2002). "Combinatorial Information Market Design"

---

## 🎉 **Summary**

You now have a production-ready system that:

✅ **Collects** Bitcoin market data every 15 minutes  
✅ **Enriches** with technical indicators & BTC spot price  
✅ **Stores** in ML-ready SQLite database  
✅ **Tracks** outcomes for supervised learning  
✅ **Trains** XGBoost models to find edges  
✅ **Predicts** live market opportunities  
✅ **Handles** errors gracefully with logging  
✅ **Scales** to track any Polymarket markets  

**Get started:**
```bash
# 1. Start collecting (leave running)
python -m polymarket_agents.services.bitcoin_tracker

# 2. Check stats after a few hours
python scripts/python/query_bitcoin_data.py --stats

# 3. Train model after a few days
python examples/train_bitcoin_predictor.py --predict-live
```

Happy trading! 🚀📈

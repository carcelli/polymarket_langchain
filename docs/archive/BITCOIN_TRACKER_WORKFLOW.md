# 🔄 Bitcoin Market Tracker - Complete Workflow

## 📊 **System Architecture**

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BITCOIN MARKET TRACKER SYSTEM                   │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  POLYMARKET  │      │   BINANCE    │      │    YOUR      │
│  Gamma API   │◄─────┤   Exchange   │◄─────┤   PYTHON     │
│              │      │              │      │   SCRIPTS    │
└──────┬───────┘      └──────┬───────┘      └──────────────┘
       │                     │
       │ GET /markets        │ GET BTC/USDT
       │ (Bitcoin)           │ (spot price)
       │                     │
       ▼                     ▼
┌─────────────────────────────────────────┐
│    BITCOIN TRACKER DAEMON               │
│  (bitcoin_tracker.py)                   │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │  Every 15 minutes:              │   │
│  │  1. Fetch BTC spot price        │   │
│  │  2. Fetch Bitcoin markets       │   │
│  │  3. Calculate indicators        │   │
│  │  4. Save snapshot to DB         │   │
│  │  5. Check resolutions           │   │
│  └─────────────────────────────────┘   │
│                                         │
│  Features:                              │
│  ✅ Graceful shutdown (Ctrl+C)         │
│  ✅ Error handling & retries           │
│  ✅ Structured logging                 │
│  ✅ Data quality scoring               │
└─────────────┬───────────────────────────┘
              │
              │ INSERT snapshots
              │
              ▼
┌──────────────────────────────────────────┐
│      SQLITE DATABASE                     │
│  (data/bitcoin_tracker.db)               │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │ market_snapshots                   │ │
│  │  - timestamp, market_id, question  │ │
│  │  - yes_price, no_price, volume     │ │
│  │  - btc_spot_price, indicators      │ │
│  │  - resolved, outcome (labels)      │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │ market_resolutions                 │ │
│  │  - outcome tracking for backfill   │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │ collection_runs                    │ │
│  │  - monitoring & health checks      │ │
│  └────────────────────────────────────┘ │
└─────┬──────────────────────────┬─────────┘
      │                          │
      │ Query                    │ Export
      │                          │
      ▼                          ▼
┌─────────────────┐    ┌──────────────────────┐
│  QUERY TOOL     │    │   ML TRAINING        │
│                 │    │                      │
│ query_bitcoin_  │    │ train_bitcoin_       │
│ data.py         │    │ predictor.py         │
│                 │    │                      │
│ • Statistics    │    │ 1. Load resolved     │
│ • Export CSV    │    │    markets           │
│ • Market history│    │ 2. Feature prep      │
│ • ML dataset    │    │ 3. Train XGBoost     │
└─────────────────┘    │ 4. Evaluate model    │
                       │ 5. Feature importance│
                       │ 6. Predict live      │
                       │ 7. Find edges        │
                       └──────────┬───────────┘
                                  │
                                  ▼
                       ┌────────────────────────┐
                       │  TRADING DECISIONS     │
                       │                        │
                       │  🎯 BUY YES (edge>5%)  │
                       │  🎯 BUY NO  (edge<-5%) │
                       │  😐 PASS    (no edge)  │
                       └────────────────────────┘
```

---

## 📋 **3-Step User Workflow**

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER WORKFLOW                            │
└─────────────────────────────────────────────────────────────────┘

STEP 1: DISCOVER MARKETS
─────────────────────────
$ python scripts/python/find_markets_to_track.py

Output:
  ✅ Found 15 Active Markets
  1. Will Bitcoin reach $170,000 by Dec 31?
     ID: 574073, Volume: $7.5M
  
  Command:
  python -m polymarket_agents.services.bitcoin_tracker \
    --market-ids 574073,12345,67890

        │
        ▼

STEP 2: COLLECT DATA (Run continuously)
────────────────────────────────────────
$ python -m polymarket_agents.services.bitcoin_tracker \
    --market-ids 574073

Output:
  📸 Collecting snapshot...
  ₿  BTC: $87,847.07 (-1.29% 24h)
  ✅ Collected 8/8 markets
  💾 Database: 128 snapshots
  😴 Sleeping 900s...
  
  [Runs 24/7, press Ctrl+C to stop]

        │
        │ (After a few days/weeks...)
        ▼

STEP 3: TRAIN & PREDICT
────────────────────────
$ python examples/train_bitcoin_predictor.py --predict-live

Output:
  📊 Training Data: 1,234 samples
  📈 ROC AUC: 0.781
  
  🎯 Found 3 Trading Opportunities:
  
  1. Will Bitcoin reach $150k by March 31?
     Market: 12% | ML: 28.5%
     Edge: +16.5%
     🎯 BUY YES (EV: 0.163)
```

---

## 🔄 **Data Flow Diagram**

```
COLLECTION PHASE (Every 15 minutes)
────────────────────────────────────

  API Calls               Processing              Storage
  ─────────               ──────────              ───────
  
  Polymarket     ──┐
  /markets         │
  (Bitcoin Q's)    ├──► Parse       ──┐
                   │    markets       │
  Binance        ──┘                  │
  BTC/USDT                            ├──► Calculate    ──► SQLite DB
  (spot price)                        │    Indicators       (snapshot)
                                      │    - Momentum
  Historical     ────────────────────►│    - RSI
  Snapshots                           │    - Volatility
  (from DB)                           │    - Volume spike
                                      └──► Assign
                                           quality score


TRAINING PHASE (After collecting data)
───────────────────────────────────────

  Database              ML Pipeline            Predictions
  ────────              ───────────            ───────────
  
  SQLite DB      ──► Load           ──► XGBoost      ──► Live
  (resolved          resolved           Classifier        Market
   markets)          snapshots                            Analysis
                      │                                   │
                      ▼                                   ▼
                   Feature         ──► Train/          ──► Edge
                   Matrix              Evaluate            Detection
                   (X)                                     │
                      │                                    ▼
                      ▼                                 Recommend:
                   Labels                               BUY/PASS
                   (y)


PREDICTION PHASE (Real-time)
─────────────────────────────

  Live Market         Model             Decision
  ───────────         ─────             ────────
  
  Current         ──► Trained       ──► ML Prob:     ──► Compare:
  Market              XGBoost           28.5%            ML vs Market
  State                                                   │
    - yes_price:                                          ▼
      12%                                              Edge: +16.5%
    - volume                                              │
    - BTC price                                           ▼
    - indicators                                       🎯 BUY YES
```

---

## 🎯 **Feature Engineering Pipeline**

```
RAW DATA                FEATURES               USE IN ML
────────                ────────               ─────────

Market Price   ──► market_probability    ──► Baseline crowd wisdom
                   (yes_price)

BTC Spot       ──► btc_spot_price        ──► Context for BTC markets
                   btc_24h_change_pct    ──► Sentiment indicator

Historical     ──► price_momentum_15m    ──► Short-term trend
Prices             price_momentum_1h     ──► Medium-term trend
                   price_volatility      ──► Uncertainty measure

Volume Data    ──► volume_spike          ──► Unusual activity
                   (vol - avg_vol)       ──► Insider info signal?

Price Series   ──► rsi_14               ──► Overbought/oversold
                   (RSI calculation)     ──► Mean reversion signal

Market Price   ──► market_edge          ──► Crowd confidence
                   abs(price - 0.5)      ──► Polarization measure

Time           ──► time_to_expiry_hours ──► Urgency factor
                                         ──► Decay modeling
```

---

## 📈 **ML Model Architecture**

```
INPUT FEATURES (12)                     OUTPUT
───────────────                         ──────

market_probability      ┐
volume                  │
liquidity               │
btc_spot_price          │               probability_yes
btc_24h_change_pct      ├─────► XGBoost ────► (0.0 to 1.0)
price_momentum_15m      │     Classifier
price_momentum_1h       │         │
volume_spike            │         │
price_volatility        │         │
rsi_14                  │         ▼
market_edge             │    Feature
time_to_expiry_hours    ┘    Importance
                                  │
                                  ▼
                          market_probability: 0.32
                          btc_24h_change_pct: 0.18
                          volume: 0.14
                          ...


EDGE CALCULATION
────────────────

ML Prediction: 28.5%  (model says 28.5% chance YES)
Market Price:  12.0%  (crowd says 12% chance YES)
                      
Edge = 28.5% - 12.0% = +16.5%

If Edge > +5%:  BUY YES  (model is bullish)
If Edge < -5%:  BUY NO   (model is bearish)
Otherwise:      PASS     (no clear signal)

Expected Value = (ML_prob × (1 - market_prob)) 
                - ((1 - ML_prob) × market_prob)
               = (0.285 × 0.88) - (0.715 × 0.12)
               = 0.163 (16.3% expected profit)
```

---

## 🔧 **System Components**

```
┌────────────────────────────────────────────────────────┐
│               SYSTEM ARCHITECTURE                      │
└────────────────────────────────────────────────────────┘

COLLECTION LAYER (bitcoin_tracker.py)
──────────────────────────────────────
  • BitcoinMarketTracker class
  • API client (httpx)
  • Exchange client (ccxt.binance)
  • Technical indicator calculations
  • Database operations (sqlite3)
  • Signal handlers (graceful shutdown)

DATA LAYER (SQLite)
───────────────────
  • market_snapshots (main table)
  • market_resolutions (outcome tracking)
  • collection_runs (monitoring)
  • Indexes for performance

QUERY LAYER (query_bitcoin_data.py)
────────────────────────────────────
  • Statistics aggregation
  • Data export (CSV/JSON)
  • ML dataset preparation
  • Market history viewer

ML LAYER (train_bitcoin_predictor.py)
──────────────────────────────────────
  • BitcoinMarketPredictor class
  • XGBoost training pipeline
  • Feature engineering
  • Cross-validation
  • Edge detection algorithm
  • Live prediction system

UTILITY LAYER (find_markets_to_track.py)
─────────────────────────────────────────
  • Market discovery
  • Filtering by category/keywords
  • Command generation
```

---

## 🎯 **Success Metrics**

```
DATA COLLECTION
───────────────
✅ Uptime: >99% (automatic restarts on failures)
✅ Data Quality: >80% of snapshots with quality ≥0.8
✅ Latency: <5s per snapshot collection
✅ Coverage: All target markets tracked every 15 min

ML MODEL
────────
✅ Accuracy: >70% on test set
✅ ROC AUC: >0.75 (good discrimination)
✅ Precision: >65% (avoid false positives)
✅ Feature Importance: market_probability top feature

TRADING PERFORMANCE (Backtesting)
──────────────────────────────────
✅ Win Rate: >55% on recommended trades
✅ Expected Value: >0.10 average per trade
✅ Sharpe Ratio: >1.5 (risk-adjusted returns)
✅ Max Drawdown: <20% of bankroll
```

---

## 🚀 **Production Deployment**

```
DEVELOPMENT → STAGING → PRODUCTION
───────────   ───────   ──────────

1. Local Testing
   • Run --once to test snapshot
   • Verify data quality
   • Check database schema

2. Extended Collection
   • Run for 1 week continuously
   • Monitor logs for errors
   • Validate data completeness

3. Model Training
   • Train on ≥100 resolved markets
   • Cross-validate performance
   • Tune hyperparameters

4. Paper Trading
   • Track recommendations
   • Simulate trades
   • Measure actual vs predicted

5. Live Deployment
   • Start with small positions
   • Monitor edge detection
   • Scale gradually
```

---

## 📚 **Quick Reference Commands**

```bash
# SETUP
pip install httpx ccxt xgboost pandas scikit-learn

# DISCOVERY
python scripts/python/find_markets_to_track.py --keywords bitcoin

# COLLECTION
python -m polymarket_agents.services.bitcoin_tracker --market-ids IDS

# MONITORING
python scripts/python/query_bitcoin_data.py --stats

# ANALYSIS
python scripts/python/query_bitcoin_data.py --market 574073

# EXPORT
python scripts/python/query_bitcoin_data.py --export csv

# TRAINING
python examples/train_bitcoin_predictor.py

# PREDICTION
python examples/train_bitcoin_predictor.py --predict-live --min-edge 0.05

# BACKGROUND RUN
nohup python -m polymarket_agents.services.bitcoin_tracker > tracker.log 2>&1 &
```

---

## 🎉 **You're Ready!**

This workflow gives you:
1. **Continuous data collection** (24/7 tracking)
2. **ML-ready features** (12+ engineered features)
3. **Automated training** (XGBoost classifier)
4. **Edge detection** (ML vs market comparison)
5. **Production monitoring** (logs, stats, quality scores)

Start collecting data today and train your first model in a few days! 🚀

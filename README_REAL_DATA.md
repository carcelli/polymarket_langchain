# 🚀 Using LangGraph Agents with Real Polymarket Data

This guide shows how to operationalize your LangGraph agents for real-world market analysis using actual Polymarket data.

## 📊 Current Data Status

Your system contains **20,716 active markets** from Polymarket with:
- ✅ **Live market data** (questions, prices, volumes)
- ✅ **Historical price data** (for trend analysis)
- ✅ **Category classification** (politics, sports, crypto, etc.)
- ✅ **Volume metrics** (trading activity)

## 🎯 Quick Start Commands

### Check Available Markets
```bash
# See how many active markets we have
python -c "
import sqlite3
conn = sqlite3.connect('data/markets.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM markets WHERE active = 1')
print(f'Active markets: {cursor.fetchone()[0]}')
conn.close()
"
```

### Analyze Specific Markets
```bash
# High-volume geopolitics market
python scripts/python/cli.py run-planning-agent "Russia x Ukraine ceasefire in 2025?"

# Find crypto opportunities
python scripts/python/cli.py run-memory-agent "Find interesting crypto markets to trade"

# Sports market analysis
python scripts/python/cli.py run-planning-agent "Will the Tennessee Titans win Super Bowl 2026?"
```

### Run Comprehensive Analysis
```bash
# Full workflow demonstration
python market_analysis_workflow.py

# Analyze specific category
python -c "
from market_analysis_workflow import MarketAnalyzer
analyzer = MarketAnalyzer()
opportunities = analyzer.find_opportunities_by_category('crypto', min_volume=1000000)
print(f'Found {len(opportunities)} crypto markets to analyze')
"
```

## 📈 Top Markets by Volume (Real Data)

| Rank | Market | Volume | Category | Implied Prob |
|------|--------|--------|----------|--------------|
| 1 | Xi Jinping out in 2025? | $75M | TECH | 0.4% |
| 2 | Tennessee Titans Super Bowl? | $71M | SPORTS | 0.0% |
| 3 | Russia-Ukraine ceasefire 2025? | $68M | GEOPOLITICS | 1.1% |
| 4 | Miami Dolphins Super Bowl? | $64M | SPORTS | 0.0% |
| 5 | New York Jets Super Bowl? | $57M | SPORTS | 0.0% |

## 🔧 Agent Capabilities with Real Data

### Memory Agent (`run-memory-agent`)
- **Local Database Queries**: Searches 20K+ markets instantly
- **Category Analysis**: Finds patterns across market types
- **Volume Filtering**: Identifies high-activity markets
- **Enrichment**: Adds live API data when needed

**Example Output:**
```
🧠 Memory Node: Querying local knowledge base...
   📂 Found 20 crypto markets
   ✅ Memory context loaded: 20,716 markets available
🔄 Enrichment Node: Checking if live data needed...
   ⏭️ Memory sufficient, skipping API call
```

### Planning Agent (`run-planning-agent`)
- **Market Intelligence**: Gathers volume, prices, trends
- **Statistical Analysis**: Calculates implied probabilities
- **Edge Detection**: Identifies mispriced opportunities
- **Kelly Sizing**: Optimal position sizing recommendations

**Example Output:**
```
📊 Planning Agent analyzing: Russia x Ukraine ceasefire in 2025?
📚 Research Node: Gathering market intelligence...
   📍 Found market: Russia x Ukraine ceasefire in 2025?...
   📊 Volume: $67,970,317
   📈 Implied Prob: 1.1%
   💡 Decision: PASS (no edge detected)
```

## 🏗️ Building Trading Strategies

### 1. **Volume-Based Scanning**
```python
# Find high-volume opportunities
from market_analysis_workflow import MarketAnalyzer

analyzer = MarketAnalyzer()
markets = analyzer.get_high_volume_markets(min_volume=10000000)
print(f"Found {len(markets)} high-volume markets")
```

### 2. **Category-Specific Analysis**
```python
# Focus on specific categories
crypto_opportunities = analyzer.find_opportunities_by_category('crypto')
politics_opportunities = analyzer.find_opportunities_by_category('politics')
```

### 3. **Automated Opportunity Reports**
```python
# Generate comprehensive reports
report = analyzer.run_comprehensive_scan(['crypto', 'politics', 'sports'])
print(report)
```

## 📊 Real-World Analysis Examples

### Geopolitics Markets
- **Russia-Ukraine Ceasefire**: $68M volume, 1.1% probability
- **China-Taiwan Conflict**: $12M volume, 0.1% probability
- **Xi Jinping Leadership**: $75M volume, 0.4% probability

### Sports Markets
- **Super Bowl Futures**: $40-70M volume each
- **NFL Team Performance**: High liquidity, low probabilities

### Crypto Markets
- **Bitcoin Price Targets**: $20-30M volume on major milestones
- **New Project Launches**: High uncertainty, volume-driven

## 🔄 Data Pipeline

### Refresh Market Data
```bash
# Update local database with latest markets
python scripts/python/refresh_markets.py --max-events 500

# Continuous monitoring
python scripts/python/refresh_markets.py --continuous --interval 300
```

### Check Data Freshness
```bash
# See when data was last updated
python -c "
import sqlite3
conn = sqlite3.connect('data/markets.db')
cursor = conn.cursor()
cursor.execute('SELECT MAX(last_updated) FROM markets')
print(f'Last updated: {cursor.fetchone()[0]}')
conn.close()
"
```

## 🎯 Next Steps for Production Use

### 1. **API Keys Setup**
```bash
# For live trading and enhanced research
cp .env.example .env
# Add: OPENAI_API_KEY, NEWSAPI_API_KEY, TAVILY_API_KEY
```

### 2. **Automated Scanning**
```bash
# Set up cron jobs for regular analysis
./scripts/bash/setup_cron_jobs.sh

# Daily opportunity reports
python market_analysis_workflow.py > daily_report_$(date +%Y%m%d).txt
```

### 3. **Risk Management**
- Start with small position sizes
- Use Kelly fraction recommendations
- Monitor for API rate limits
- Implement circuit breakers

### 4. **Performance Tracking**
```python
# Log all analyses and decisions
# Track which signals perform well over time
# A/B test different analysis strategies
```

## 🚨 Production Considerations

- **Rate Limits**: Respect Polymarket API limits
- **Data Quality**: Validate market data before trading
- **Position Sizing**: Never risk more than you can lose
- **Market Hours**: Be aware of market open/close times
- **Regulatory**: Ensure compliance with local laws

## 📞 Getting Help

- Check agent logs in `logs/` directory
- View graph visualizations with `python visualize_graphs.py`
- Test individual components with CLI tools
- Monitor system health with `python scripts/validate_graphs.py`

---

**Your LangGraph agents are now ready for real-world Polymarket analysis!** 🎯📈

Start with the quick commands above, then scale up to automated scanning and reporting.

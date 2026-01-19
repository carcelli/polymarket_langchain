# 🚀 Quick Start: Betting on Polymarket

## ✅ Current Setup Status

- ✅ Wallet configured
- ✅ Polymarket client working
- ✅ 11,792 markets in local database
- ⚠️ Wallet balance: $0.00 (needs funding)

**Your Wallet Address**: `0xa59Dd0c1Ff78cC7Ba899De496ea1Fb82B60B1E67`

## 📝 Essential Commands (zsh-safe)

### 1. Check Your Wallet
```bash
python scripts/check_wallet.py
```

### 2. Find Bitcoin Markets
```bash
python scripts/find_bitcoin_markets.py
```

### 3. Find "Up or Down" Markets (5-15 minute bets)
```bash
# Search once
python scripts/find_updown_markets.py

# Monitor continuously (recommended)
python scripts/monitor_updown_markets.py
```

### 4. Analyze a Market (AI-powered)
```bash
python -m polymarket_agents.graph.planning_agent "Should I bet on Bitcoin reaching 140k by December 2025?"
```

### 5. Place a Bet (once funded)

**For longer-term markets:**
```bash
# Syntax: python scripts/quick_bet.py <market_id> <outcome_index> <amount_usd>

# Example: Bet $5 on YES (index 0) for Bitcoin reaching $140k
python scripts/quick_bet.py 574072 0 5.0
```

**For Up/Down markets (5-15 minutes):**
```bash
# Syntax: python scripts/quick_bet_updown.py <market_id> <UP|DOWN> <amount_usd>

# Example: Bet $20 that Bitcoin goes UP in next 15 minutes
python scripts/quick_bet_updown.py 1234567 UP 20.0
```

## 💰 Fund Your Wallet

**Send USDC (Polygon network) to:**
```
0xa59Dd0c1Ff78cC7Ba899De496ea1Fb82B60B1E67
```

**Quick funding options:**
1. **Bridge from Ethereum**: https://wallet.polygon.technology/
2. **Centralized exchange**: Binance, Coinbase (withdraw to Polygon)
3. **Buy with card**: Transak, MoonPay

**Recommended amount**: $10-50 for testing

**Verify on blockchain**: https://polygonscan.com/address/0xa59Dd0c1Ff78cC7Ba899De496ea1Fb82B60B1E67

## 🎯 Available Bitcoin Markets

| Market | Volume | Current Price |
|--------|--------|---------------|
| Bitcoin reach $170k by Dec 2025? | $7.4M | 0.05% YES |
| Bitcoin reach $140k by Dec 2025? | $6.7M | 0.15% YES |
| Bitcoin hit $1M before GTA VI? | $2.4M | 48.5% YES |
| Bitcoin reach $125k by Dec 2025? | $775k | 100% YES |
| Bitcoin $80k or $150k first? | $733k | 82.6% $80k |

## ⚠️ About 15-Minute Markets

**Polymarket doesn't offer ultra-short 15-minute Bitcoin markets** because:
- Market resolution requires time for oracle verification
- Very short durations have poor liquidity
- Crypto markets typically run for days/weeks/months

**Typical market durations:**
- Hours: Rare (major breaking news events)
- Days to weeks: Common
- Months to years: Most crypto markets

## 🔍 Advanced Search

### Search by keyword
```bash
# Search for any keyword
python -c 'from polymarket_agents.memory.manager import MemoryManager; m=MemoryManager("data/markets.db"); print([mk["question"] for mk in m.search_markets("trump", limit=5)])'
```

### Get top volume markets
```bash
python -c 'from polymarket_agents.memory.manager import MemoryManager; m=MemoryManager("data/markets.db"); print([mk["question"] for mk in m.list_top_volume_markets(limit=5)])'
```

### Search by category
```bash
python -c 'from polymarket_agents.memory.manager import MemoryManager; m=MemoryManager("data/markets.db"); print([mk["question"] for mk in m.list_markets_by_category("crypto", limit=5)])'
```

## 🤖 AI Agent Commands

### Memory Agent (fast local search)
```bash
python -m polymarket_agents.graph.memory_agent "Find high volume crypto markets"
```

### Planning Agent (quantitative analysis)
```bash
python -m polymarket_agents.graph.planning_agent "Should I bet on Bitcoin reaching new ATH by March 2026?"
```

## 📊 Virtual Trading Systems (REQUIRED FIRST)

⚠️ **DO NOT fund your wallet yet!** Run virtual trading to prove edge.

### 🏀 **Sports Focus** (⭐ Recommended Start)

#### NBA Simulator
```bash
# Best for beginners - clearest signal, proven inefficiencies
python scripts/nba_simulator.py

# Conservative: 7% edge, $75k min volume, $15 bets
python scripts/nba_simulator.py 0.07 75000 15.0 300
```

**Why start here:**
- ✅ Log5 baseline showing **36% edge** opportunities!
- ✅ Longer horizons (hours → days, not minutes)
- ✅ Fundamentals matter (records, venue, rest)
- ✅ Easy to enhance (injuries, Elo, player matchups)

**Files:**
- `scripts/nba_market_fetcher.py` - Discover NBA markets
- `scripts/nba_predictor.py` - Log5 + home advantage
- `scripts/nba_simulator.py` - Paper trading system

### 🪙 **Crypto Focus** (High Frequency)

#### Crypto 15M Simulator
```bash
# Ultra-short 5-15 min crypto markets
python scripts/monitor_simulator.py

# Custom: 55% confidence, 3% risk, 30s poll
python scripts/monitor_simulator.py 0.55 0.03 30
```

**Use for:**
- ⏱️ Testing execution speed
- 🔬 Rapid iteration on signals
- ⚠️ Harder to profit (noise dominates short timeframes)

**Files:**
- `scripts/crypto_market_fetcher.py` - Discover crypto Up/Down markets
- `scripts/crypto_predictor.py` - Momentum + RSI + volume
- `scripts/monitor_simulator.py` - Paper trading system

### 🎯 **Unified System** (Multi-Market Production)

#### Virtual Trader
```bash
# Auto-selects best strategy per market type
python scripts/virtual_trader.py

# NBA + crypto markets with 3% edge threshold
python scripts/virtual_trader.py --markets nba crypto --min-edge 0.03

# NBA only (recommended after proving edge)
python scripts/virtual_trader.py --markets nba --min-edge 0.05
```

**Features:**
- ✅ Auto strategy selection (NBA → Log5, Crypto → Technical)
- ✅ Planning agent integration (future)
- ✅ ML strategy registry
- ✅ Kelly criterion position sizing
- ✅ Risk management (drawdown limits, circuit breakers)
- ✅ Performance tracking by market type

### **Check Performance**
```bash
# NBA
sqlite3 data/nba_simulator.db "SELECT COUNT(*), SUM(virtual_profit) FROM nba_bets WHERE resolved=TRUE"

# Virtual Trader
sqlite3 data/virtual_trader.db "SELECT market_type, COUNT(*), SUM(profit) FROM virtual_trades WHERE resolved=TRUE GROUP BY market_type"

# Crypto
sqlite3 data/simulator.db "SELECT SUM(virtual_profit) FROM trades WHERE resolved=TRUE"
```

### **Success Criteria Before Funding**
**NBA (20+ games minimum):**
- ✅ Win rate > 52%
- ✅ Positive P&L
- ✅ Max drawdown < 30%

**Target (50+ bets across all systems):**
- ✅ Win rate > 55%
- ✅ ROI > 5%
- ✅ Max drawdown < 20%
- ✅ Sharpe ratio > 0.5

See `docs/VIRTUAL_TRADING_GUIDE.md` for complete comparison.

## 📊 Workflow: Live Trading (After proving edge)

1. **Fund wallet** (see above) → Wait for confirmation on PolygonScan

2. **Check balance**:
   ```bash
   python scripts/check_wallet.py
   ```

3. **Find interesting market**:
   ```bash
   python scripts/find_bitcoin_markets.py
   ```

4. **Get AI analysis** (optional but recommended):
   ```bash
   python -m polymarket_agents.graph.planning_agent "Analyze market 574072 - Bitcoin reaching 140k by Dec 2025"
   ```

5. **Place small test bet**:
   ```bash
   # Bet $5 on YES
   python scripts/quick_bet.py 574072 0 5.0
   ```

6. **Monitor your position**:
   ```bash
   python -c 'from polymarket_agents.langchain.clob_tools import clob_get_open_orders; print(clob_get_open_orders())'
   ```

## 🔐 Security Reminders

- ✅ Never commit `.env` to git (already in `.gitignore`)
- ✅ Start with small amounts ($5-10) for testing
- ✅ Your private key is saved in `.env` - keep it secure
- ✅ For production, consider hardware wallets or key management services

## 🆘 Troubleshooting

### "zsh: bad substitution"
Use single quotes `'...'` instead of double quotes `"..."` in shell commands, or use the provided Python scripts instead.

### "Insufficient funds"
Your wallet has no USDC. Fund it via the address above.

### "Market not found"
Refresh the database:
```bash
python scripts/python/refresh_markets.py --max-events 500
```

### "Client not initialized"
Check that `POLYGON_WALLET_PRIVATE_KEY` in `.env` is a 64-character hex string.

## 📚 Additional Resources

- **Wallet Setup Guide**: `docs/WALLET_SETUP_GUIDE.md`
- **Full Documentation**: `CLAUDE.md`
- **Polymarket Help**: https://polymarket.com/help
- **API Reference**: `docs/POLYMARKET_API_REFERENCE.md`

## 🎉 You're Ready!

Once you fund your wallet, you're **100% ready to trade** on any of the 11,792+ markets in your database!

**Happy betting! 🎲**

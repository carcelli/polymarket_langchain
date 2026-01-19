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

## 📊 Workflow: Paper Trading First! (RECOMMENDED)

⚠️ **DO NOT fund your wallet yet!** Paper trade first to prove edge.

1. **Start automated paper trading**:
   ```bash
   python scripts/auto_paper_trader.py
   ```
   This monitors for markets, makes predictions, and places virtual bets automatically.

2. **Check performance after a week**:
   ```bash
   python scripts/paper_trading_system.py summary
   ```

3. **Run backtests**:
   ```bash
   python scripts/backtest_updown.py
   ```

4. **Only fund wallet after 200+ successful paper trades** with:
   - Win rate > 55%
   - Positive total P&L
   - Max drawdown < 20%

See `docs/PAPER_TRADING_GUIDE.md` for complete details.

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

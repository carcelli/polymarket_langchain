# 📈📉 Up or Down Markets Guide

## What Are "Up or Down" Markets?

These are **ultra-short-term binary options** on crypto prices (Bitcoin, Ethereum, Solana, XRP):

- **Duration**: 5-15 minutes
- **Question**: Will the price be HIGHER or LOWER than the starting price?
- **Starting Price**: Fixed at market open (e.g., $92,994.26)
- **Settlement**: Automatic at expiry based on price oracle

### Example Market

```
Bitcoin Up or Down
January 19, 5:00-5:15AM ET

Starting Price: $92,994.26
Current Price: $92,914

Bet: UP or DOWN?
```

If Bitcoin is **above $92,994.26** at 5:15 AM ET → UP wins  
If Bitcoin is **below $92,994.26** at 5:15 AM ET → DOWN wins

## 🔍 Finding Live Markets

### Method 1: Search Once
```bash
python scripts/find_updown_markets.py
```

Shows all Up/Down markets expiring in the next hour.

### Method 2: Continuous Monitor (Recommended)
```bash
# Check every 30 seconds (default)
python scripts/monitor_updown_markets.py

# Check every 10 seconds (more frequent)
python scripts/monitor_updown_markets.py 10

# Check every 60 seconds (less frequent)
python scripts/monitor_updown_markets.py 60
```

Runs continuously and alerts you when new markets appear.

## 🎯 Placing Bets

### Quick Bet Command
```bash
python scripts/quick_bet_updown.py <market_id> <UP|DOWN> <amount_usd>
```

### Examples

```bash
# Bet $10 that Bitcoin will go UP
python scripts/quick_bet_updown.py 1163212 UP 10.0

# Bet $25 that Ethereum will go DOWN
python scripts/quick_bet_updown.py 1163210 DOWN 25.0

# Bet $5 that Solana will go UP
python scripts/quick_bet_updown.py 1163211 UP 5.0
```

## ⚡ Complete Workflow

### Step 1: Start the Monitor
```bash
# In one terminal window
python scripts/monitor_updown_markets.py
```

Leave this running to watch for new markets.

### Step 2: When a Market Appears
```
🚨 [04:55:03] NEW MARKETS FOUND!

✅ Bitcoin Up or Down - January 19, 5-5:15AM ET
   ID: 1234567
   ⏰ Expires in: 19.8 minutes
   💰 Volume: $12,543.00
   🎯 UP:   0xabc123...
   🎯 DOWN: 0xdef456...

💡 To bet: python scripts/quick_bet_updown.py 1234567 UP 10.0
```

### Step 3: Decide and Execute
```bash
# Quick decision - bet $20 on UP
python scripts/quick_bet_updown.py 1234567 UP 20.0
```

### Step 4: Wait for Expiry
- Market automatically resolves at expiry time
- Winning side gets paid out
- Check your balance afterward

## 📊 Strategy Considerations

### Time-Based Patterns
- **Volume**: Higher volume = more liquid, tighter spreads
- **Time of day**: US market hours (9 AM - 4 PM ET) have highest activity
- **Expiry timing**: 5-minute windows are more volatile than 15-minute

### Quick Analysis
```bash
# Get AI analysis before betting
python -m polymarket_agents.graph.planning_agent "Bitcoin just broke $93k resistance. Should I bet UP on the next 15-minute window?"
```

### Risk Management
- **Start small**: $5-10 per bet until you understand dynamics
- **Time decay**: Don't bet too close to expiry (< 2 minutes)
- **Spread risk**: Check orderbook depth before large bets
- **Diversify**: Don't put all funds in one 15-minute window

## 🕐 Market Schedule

Up/Down markets typically appear:
- **US Trading Hours**: 9 AM - 4 PM ET (most active)
- **Frequency**: Every 5-15 minutes during active hours
- **Cryptocurrencies**: Bitcoin, Ethereum, Solana, XRP
- **Windows**: 5-minute and 15-minute durations

**Note**: Not all windows have markets - depends on liquidity/demand.

## 💰 Payout Calculation

### Winning Bet
```
If price moves in your favor:
- You receive $1.00 per share
- Profit = (1.00 - purchase_price) × shares

Example:
- Bought 20 shares of UP at $0.55 each = $11 cost
- Bitcoin goes UP at expiry
- Payout: 20 × $1.00 = $20
- Profit: $20 - $11 = $9
```

### Losing Bet
```
If price moves against you:
- Shares become worthless
- Loss = your initial investment

Example:
- Bought 20 shares of UP at $0.55 each = $11 cost
- Bitcoin goes DOWN at expiry
- Payout: $0
- Loss: $11
```

## 🚨 Important Warnings

### Time Sensitivity
⚠️ **These markets expire FAST!**
- Don't walk away after placing a bet
- Set a timer for the expiry time
- Markets can't be exited early (no secondary market typically)

### Price Discovery Period
⚠️ **First 1-2 minutes after open:**
- Orderbook is thin
- Spreads are wide
- Prices are discovering fair value
- **Wait 30-60 seconds** before betting for better prices

### Network Delays
⚠️ **Polygon network:**
- Transactions take 2-5 seconds to confirm
- During high volatility, gas prices spike
- Leave buffer before expiry (don't bet with < 1 minute left)

### Liquidity Risk
⚠️ **Low volume markets:**
- Harder to get fills at desired prices
- May need to use market orders (instant but worse price)
- Stick to markets with $5k+ volume

## 🔧 Troubleshooting

### "No markets found"
These markets are time-specific. Try:
- Running during US market hours (9 AM - 4 PM ET)
- Checking back in 5-10 minutes
- Running the monitor script continuously

### "Order rejected"
Possible causes:
- Insufficient balance
- Market expired while placing order
- Network congestion
- Token ID incorrect

### "Token ID not found"
The market structure changed. Try:
- Re-fetching market data
- Using the quick_bet_updown.py script (auto-detects tokens)
- Manual inspection: `python -c 'import requests; print(requests.get("https://gamma-api.polymarket.com/markets/<id>").json())'`

## 📈 Advanced: Automated Trading

### Simple Bot Structure
```python
# WARNING: Use at your own risk!

from scripts.monitor_updown_markets import get_updown_markets
from scripts.quick_bet_updown import get_market_from_gamma
import time

while True:
    markets = get_updown_markets()
    
    for market in markets:
        # Add your strategy logic here
        # Example: Simple momentum strategy
        if is_bullish_signal(market):
            execute_bet(market['id'], 'UP', 10.0)
        
    time.sleep(30)
```

### Risk Controls
- **Max bet size**: Never risk more than 1-2% of bankroll per trade
- **Daily loss limit**: Stop if down X% for the day
- **Win/loss tracking**: Log all trades for analysis
- **Circuit breaker**: Auto-stop if 5 consecutive losses

## 🎓 Learning Resources

### Practice First
1. **Paper trading**: Track hypothetical bets for a week
2. **Micro bets**: Start with $1-2 per bet
3. **Review**: Analyze what worked and what didn't

### Market Dynamics
- Watch orderbook depth before betting
- Understand implied probability (price = probability)
- Learn to read momentum indicators
- Practice timing entries (not too early, not too late)

## 📞 Support

- **Script issues**: Check logs and error messages
- **Wallet problems**: See `docs/WALLET_SETUP_GUIDE.md`
- **Market questions**: See `QUICK_START.md`
- **Polymarket Help**: https://polymarket.com/help

---

**Remember**: These are speculative bets on short-term price movements. Only risk what you can afford to lose!

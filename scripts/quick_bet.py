#!/usr/bin/env python3
"""
Quick betting script for Polymarket markets.

Usage:
    python scripts/quick_bet.py <market_id> <outcome_index> <amount_usd>

Example:
    python scripts/quick_bet.py 574072 0 5.0
    (Bet $5 on outcome 0 (YES) for market 574072)
"""

import sys
from polymarket_agents.memory.manager import MemoryManager
from polymarket_agents.connectors.polymarket import Polymarket

def main():
    if len(sys.argv) < 4:
        print("Usage: python scripts/quick_bet.py <market_id> <outcome_index> <amount_usd>")
        print("\nExample: python scripts/quick_bet.py 574072 0 5.0")
        print("         (Bet $5 on outcome 0 for market 574072)")
        sys.exit(1)
    
    market_id = sys.argv[1]
    outcome_index = int(sys.argv[2])
    amount_usd = float(sys.argv[3])
    
    print(f"\n🎲 Preparing to bet ${amount_usd:.2f} on market {market_id}...\n")
    
    # Get market details
    memory = MemoryManager("data/markets.db")
    market = memory.get_market(market_id)
    
    if not market:
        print(f"❌ Market {market_id} not found in database")
        sys.exit(1)
    
    print("=" * 70)
    print(f"Market: {market['question']}")
    print(f"Outcomes: {market.get('outcomes', 'N/A')}")
    print(f"Current Prices: {market.get('outcome_prices', 'N/A')}")
    print("=" * 70)
    
    # Get token ID for the outcome
    token_ids = market.get('clob_token_ids')
    if not token_ids or len(token_ids) <= outcome_index:
        print(f"❌ Invalid outcome index {outcome_index}")
        print(f"   Available outcomes: {market.get('outcomes', 'N/A')}")
        sys.exit(1)
    
    token_id = token_ids[outcome_index] if isinstance(token_ids, list) else token_ids.split(',')[outcome_index]
    outcome_name = market['outcomes'][outcome_index] if isinstance(market['outcomes'], list) else market['outcomes'].split(',')[outcome_index]
    
    print(f"\n📊 Betting Details:")
    print(f"   Outcome: {outcome_name}")
    print(f"   Token ID: {token_id}")
    print(f"   Amount: ${amount_usd:.2f} USDC")
    
    # Check balance
    poly = Polymarket()
    balance = poly.get_usdc_balance()
    
    print(f"\n💰 Wallet Balance: ${balance:.2f} USDC")
    
    if balance < amount_usd:
        print(f"\n❌ Insufficient funds! You need ${amount_usd:.2f} but only have ${balance:.2f}")
        print(f"   Fund your wallet: 0xa59Dd0c1Ff78cC7Ba899De496ea1Fb82B60B1E67")
        sys.exit(1)
    
    # Confirm
    confirm = input(f"\n⚠️  Execute BET ${amount_usd:.2f} on '{outcome_name}'? (yes/no): ").strip().lower()
    
    if confirm not in ['yes', 'y']:
        print("❌ Bet cancelled.")
        sys.exit(0)
    
    print("\n🚀 Placing market order...")
    
    try:
        from polymarket_agents.tools.trade_tools import _execute_market_order_impl
        
        result = _execute_market_order_impl(
            token_id=token_id,
            amount=amount_usd,
            side="BUY"
        )
        
        print("\n✅ Order executed!")
        print(result)
        
    except Exception as e:
        print(f"\n❌ Error placing bet: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()

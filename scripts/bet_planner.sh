#!/bin/bash
# Polymarket Bet Planning Agent
# 
# Usage:
#   ./bet_planner.sh "Bitcoin 100k"        # Analyze specific market
#   ./bet_planner.sh --scan                # Scan all categories
#   ./bet_planner.sh --scan politics       # Scan specific category
#   ./bet_planner.sh --portfolio           # View portfolio

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"

if [ "$1" == "--portfolio" ]; then
    python -c "
from polymarket_agents.memory.manager import MemoryManager
mm = MemoryManager('data/markets.db')
summary = mm.get_portfolio_summary()
print()
print('📊 PORTFOLIO SUMMARY')
print('=' * 40)
print(f'Open Positions:   {summary[\"open_positions\"]}')
print(f'Total Invested:   \${summary[\"total_invested\"]:,.2f}')
print(f'Current Value:    \${summary[\"current_value\"]:,.2f}')
print(f'Unrealized P&L:   \${summary[\"unrealized_pnl\"]:,.2f}')
print(f'Realized P&L:     \${summary[\"realized_pnl\"]:,.2f}')
print(f'Total P&L:        \${summary[\"total_pnl\"]:,.2f}')
print('=' * 40)
"
elif [ "$1" == "--scan" ]; then
    category="${2:-}"
    python -m polymarket_agents.graph.planning_agent --scan $category
else
    query="$*"
    if [ -z "$query" ]; then
        echo "Usage:"
        echo "  $0 \"Market question\"    # Analyze a market"
        echo "  $0 --scan               # Find value opportunities"
        echo "  $0 --scan politics      # Scan category"
        echo "  $0 --portfolio          # View positions"
        exit 1
    fi
    python -m polymarket_agents.graph.planning_agent "$query"
fi


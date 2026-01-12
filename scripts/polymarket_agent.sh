#!/bin/bash
# Polymarket Memory Agent CLI
# 
# Usage:
#   ./scripts/polymarket_agent.sh "Your query here"
#
# Examples:
#   ./scripts/polymarket_agent.sh "What are the best Trump markets?"
#   ./scripts/polymarket_agent.sh "Find crypto arbitrage opportunities"
#   ./scripts/polymarket_agent.sh "Which NFL team should I bet on?"

cd "$(dirname "$0")/.."
PYTHONPATH=src:. python -m polymarket_agents.graph.memory_agent "$@"


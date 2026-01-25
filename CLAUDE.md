# CLAUDE.md

Polymarket prediction market agents using LangChain/LangGraph.

## Setup

```bash
pip install -e .
cp .env.example .env  # Add OPENAI_API_KEY
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python scripts/python/refresh_markets.py --max-events 200
```

## Run

```bash
# Domain agents (new architecture)
python -m polymarket_agents.domains.crypto.agent
python -m polymarket_agents.domains.nba.agent --mode games

# Legacy agents
python -m polymarket_agents.graph.memory_agent "Find crypto markets"
python -m polymarket_agents.graph.planning_agent "Will BTC hit 100k?"

# Tests
python -m pytest
```

## Architecture

```
src/polymarket_agents/
├── domains/           # Domain-specific agents (crypto, nba)
│   ├── registry.py    # Plugin system - register new domains here
│   ├── crypto/        # Binary price prediction markets
│   └── nba/           # Game outcomes + player props
├── langchain/
│   ├── tools.py       # LangChain tools
│   └── domain_tools.py # Bridge: domains → LangChain tools
├── connectors/        # API clients (gamma.py, polymarket.py)
├── graph/             # LangGraph agents (memory_agent, planning_agent)
└── context.py         # Dependency injection
```

## Adding a Domain

```python
from polymarket_agents.domains import register_domain, DomainConfig

register_domain(DomainConfig(
    name="politics",
    description="Political prediction markets",
    agent_factory=lambda ctx: PoliticsAgent(),
    scanner_factory=lambda ctx: PoliticsScanner(),
))

# Now available as LangChain tool automatically
from polymarket_agents.langchain.tools import get_domain_tools
tools = get_domain_tools("politics")
```

## Environment Variables

- `OPENAI_API_KEY` - Required
- `POLYGON_WALLET_PRIVATE_KEY` - For trading (optional)
- `DATABASE_PATH` - SQLite path (default: data/markets.db)

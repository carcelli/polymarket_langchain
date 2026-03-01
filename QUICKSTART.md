# Quickstart

Polymarket prediction market agents using LangChain + LangGraph.

## 1. Install

```bash
# Clone & install (editable + dev tools)
pip install -e ".[dev]"

# Or with uv (faster)
uv pip install -e ".[dev]"
```

## 2. Configure

```bash
cp .env.example .env
# Required: OPENAI_API_KEY
# Optional: POLYGON_WALLET_PRIVATE_KEY (for trading)
```

## 3. Seed local database

Fetches markets from the Gamma API into `data/markets.db`:

```bash
python scripts/python/refresh_markets.py --max-events 200
```

For continuous refresh every 5 minutes:

```bash
python scripts/python/refresh_markets.py --continuous --interval 300
```

## 4. Run agents

```bash
# Crypto domain agent (scans BTC/ETH/SOL price markets)
python -m polymarket_agents.domains.crypto.agent

# NBA domain agent
python -m polymarket_agents.domains.nba.agent --mode games

# Memory agent (DB-backed, general queries)
python -m polymarket_agents.graph.memory_agent "Find the most liquid crypto markets"

# Planning agent (edge + Kelly sizing)
python -m polymarket_agents.graph.planning_agent "Will BTC hit 100k by end of 2025?"
```

## 5. Run tests

```bash
python -m pytest tests/ -x -q
```

## 6. Add a new domain

```python
from polymarket_agents.domains import register_domain, DomainConfig

register_domain(DomainConfig(
    name="politics",
    description="Political prediction markets",
    agent_factory=lambda ctx: PoliticsAgent(),
    scanner_factory=lambda ctx: PoliticsScanner(),
))

# Automatically available as a LangChain tool
from polymarket_agents.langchain.domain_tools import get_domain_tools
tools = get_domain_tools("politics")
```

## 7. Docker (for deployment)

```bash
docker build -t polymarket-agents .
docker run --env-file .env polymarket-agents
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | LLM inference |
| `POLYGON_WALLET_PRIVATE_KEY` | No | Live trading on Polygon |
| `TAVILY_API_KEY` | No | Web search enrichment |
| `NEWSAPI_API_KEY` | No | News enrichment |
| `DATABASE_PATH` | No | SQLite path (default: `data/markets.db`) |

## Architecture

```
refresh_markets.py → Gamma API → data/markets.db
                                       ↓
                              MemoryManager (SQLite)
                                       ↓
              graph/memory_agent  ←───┤
              graph/planning_agent←───┘
                       ↓
              langchain/tools.py (30+ tools)
              langchain/domain_tools.py (plugin bridge)
                       ↓
              domains/crypto/   domains/nba/   (+ any new domain)
```

See `ARCHITECTURE.md` for full design and `CLAUDE.md` for agent development guidelines.

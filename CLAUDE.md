# CLAUDE.md

Polymarket prediction market agents using LangChain/LangGraph.

## Python Environment

**Always use the `agent-dev` conda environment** — the system Python lacks deps.

```bash
conda activate agent-dev        # Python 3.12.1, all deps installed
# or use the full path:
/home/orson-dev/miniconda/envs/agent-dev/bin/python
```

## Setup (first time)

```bash
conda activate agent-dev
pip install -e ".[dev]"                                    # install package
cp .env.example .env                                       # add OPENAI_API_KEY
python scripts/python/refresh_markets.py --max-events 500 # seed DB (~476 markets)
python scripts/python/bootstrap_xgboost.py                 # train baseline XGBoost model
```

## Run

```bash
# E2E dry run (no API key, no network, fully offline)
python scripts/python/e2e_dryrun.py --mock

# Domain agents
python -m polymarket_agents.domains.crypto.agent
python -m polymarket_agents.domains.nba.agent --mode games

# Graph agents
python -m polymarket_agents.graph.memory_agent "Find crypto markets"
python -m polymarket_agents.graph.planning_agent "Will BTC hit 100k?"

# Tests (142/142 pass)
python -m pytest tests/ -x -q
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

- `OPENAI_API_KEY` - Required for planning_agent probability_node (LLM reasoning)
- `POLYGON_WALLET_PRIVATE_KEY` - Required for live order execution (optional for dry run)
- `DATABASE_PATH` - SQLite path (default: data/markets.db)

## Current Status (2026-03-01)

| Layer | Status | Notes |
|-------|--------|-------|
| Package install | ✅ | `pip install -e ".[dev]"` clean |
| Test suite | ✅ | 142/142 pass (conda env) |
| Web3 v7 + eth-account | ✅ | Compatible; OrderData uses string amounts |
| Pydantic v2 compat | ✅ | No pydantic_v1 imports remain |
| GammaMarketsInput schema | ✅ | args_schema: Type[...] properly annotated |
| Market DB | ✅ | 476 real markets seeded (Mar 2026) |
| XGBoost baseline model | ✅ | data/models/xgboost_probability_model.json |
| E2E dry run | ✅ | scripts/python/e2e_dryrun.py --mock |
| Live OPENAI inference | ⚠️ | Needs OPENAI_API_KEY in .env |
| Live order broadcast | ⚠️ | Needs POLYGON_WALLET_PRIVATE_KEY |
| Continuous DB refresh | ⚠️ | Run: refresh_markets.py --continuous --interval 300 |

## Next Phase: Real Trading

1. Add `OPENAI_API_KEY` to `.env`
2. Run `python -m polymarket_agents.graph.planning_agent "Bitcoin" --scan crypto`
3. Verify planning agent produces BET/WATCH recommendations
4. Add `POLYGON_WALLET_PRIVATE_KEY` to `.env` for live execution
5. Replace dummy private key in `e2e_dryrun.py` with real key to test a signed broadcast

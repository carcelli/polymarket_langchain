# 🧠 Polymarket LangChain Agent

Build research, forecasting, and trading workflows on Polymarket using LangChain + LangGraph. ⚡

## ⚡ TL;DR
```bash
conda env create -f environment.yml
conda activate polymarket-agent
pip install -e ".[dev]"
cp .env.example .env
python scripts/python/refresh_markets.py --max-events 200
python scripts/python/cli.py run-memory-agent "Find interesting markets"
```
> 💡 Add API keys in `.env` for live calls and trading.

## ✨ What You Can Do
- 🔎 Discover markets and events via Gamma + local DB
- 🧠 Analyze value with LangGraph (edge, EV, Kelly)
- 📰 Enrich research with NewsAPI + Tavily search
- 📚 Build local RAG indexes with ChromaDB
- 💸 Place CLOB orders (limit/market) with py_clob_client
- 🛠️ Automate ingestion, refresh, and monitoring pipelines

## 🧭 Architecture at a glance
```
Gamma API ──▶ Refresh/Category Workflows ──▶ SQLite (markets.db)
                                            └──▶ LangGraph Agents (Memory + Planning)
NewsAPI/Tavily ─────────────────────────────▶ Research Tools
CLOB API ◀──────────────────────────────────▶ Trading Tools
```

## 🗂️ Repository layout
- 🧠 `src/polymarket_agents/` core library (Package: `polymarket_agents`)
  - 🔌 `src/polymarket_agents/connectors/` API clients & Integrations (`polymarket.py`, `gamma.py`, `news.py`, `chroma.py`)
  - 🧰 `src/polymarket_agents/langchain/` LangChain tools + helpers (`tools.py`, `clob_tools.py`)
  - 🧭 `src/polymarket_agents/graph/` LangGraph agents + state (`memory_agent.py`, `planning_agent.py`, `state.py`)
  - 🧪 `src/polymarket_agents/application/` workflows (`executor.py`, `creator.py`)
  - 🧩 `src/polymarket_agents/tools/` tool wrappers (`market_tools.py`, `trade_tools.py`, `research_tools.py`)
  - 🗃️ `src/polymarket_agents/memory/` Memory management (`manager.py`)
  - 🤖 `src/polymarket_agents/automl/` AutoML & Data Quality pipeline
  - 🤖 `src/polymarket_agents/subagents/` Specialized Sub-agents (GitHub, ML, Risk)
  - 🧱 `src/polymarket_agents/utils/` utilities (`config.py`, `objects.py`)
- 🧰 `scripts/` runnable entry points (see scripts index below)
- 🧪 `tests/` pytest/unittest coverage
- 📚 `docs/` deep reference docs + examples
- 🗃️ `data/` Local data (ignored by git)
- 📝 `logs/` runtime logs
- 🧬 `langgraph.json` LangGraph config
- ⚡ `examples/fetch_active_bets.py` quick Gamma API sample
- 🐳 `Dockerfile`, `environment.yml`, `requirements.txt`, `pyproject.toml`
- 🧹 `.langgraph_api/` local LangGraph runtime artifacts
- 🙌 `CONTRIBUTING.md`, `LICENSE.md`, `CHANGELOG.md`

## 🚀 Quick start

See [QUICKSTART.md](QUICKSTART.md) for the full guide. In short:

```bash
pip install -e ".[dev]"
cp .env.example .env    # add OPENAI_API_KEY
```

4) Seed the local DB (recommended):
```bash
python scripts/python/refresh_markets.py --max-events 200
```

## ⚙️ Configuration (.env)
Required for LLM + embeddings:
- `OPENAI_API_KEY`

Required for trading:
- `POLYGON_WALLET_PRIVATE_KEY` (or `PK`)

Optional connectors and tools:
- `NEWSAPI_API_KEY` (NewsAPI)
- `TAVILY_API_KEY` (Tavily search)
- `ANTHROPIC_API_KEY` (deep research agent, if present)
- `CLOB_API_URL` (default: https://clob.polymarket.com)
- `CHAIN_ID` (default: 137 / Polygon)
- `CLOB_API_KEY`, `CLOB_SECRET`, `CLOB_PASS_PHRASE` (skip derivation)

> 🔐 Keep `.env` private. Never commit keys.

## 🧰 Usage

### 🧪 CLI (Typer)
Run:
```bash
python scripts/python/cli.py --help
```

Popular commands:
- `get-all-markets`, `get-all-events`
- `get-relevant-news`
- `create-local-markets-rag`, `query-local-markets-rag`
- `ask-superforecaster`, `create-market`
- `ask-llm`, `ask-polymarket-llm`
- `run-autonomous-trader`
- `run-memory-agent`, `run-planning-agent`
- `scan-opportunities`, `list-agents`
- `run-deep-research-agent` (requires `agents/deep_research_agent.py` + ANTHROPIC/TAVILY keys)

Examples:
```bash
python scripts/python/cli.py get-all-markets --limit 5
python scripts/python/cli.py get-relevant-news "market keywords"
python scripts/python/cli.py run-memory-agent "Find interesting political markets"
python scripts/python/cli.py run-planning-agent "Will BTC hit 100k?"
python scripts/python/cli.py scan-opportunities --category politics
```

### 🧠 LangGraph agents
- 🧳 Memory Agent: local-first market analysis + API enrichment.
- 📊 Planning Agent: implied probability, EV, and Kelly sizing.
- 🔍 Opportunity Scanner: batch scan for positive edge.

LangGraph config lives in `langgraph.json` and wires:
- `memory_agent` to `polymarket_agents/graph/memory_agent.py:create_memory_agent`
- `planning_agent` to `polymarket_agents/graph/planning_agent.py:create_planning_agent`

Direct runs:
```bash
python -m polymarket_agents.graph.memory_agent "What are the top crypto markets?"
python -m polymarket_agents.graph.planning_agent "Will the Fed cut rates in Q1 2025?"
python -m polymarket_agents.graph.planning_agent --scan politics
```

### 🧑‍💻 Python usage
```python
from polymarket_agents.langchain.agent import create_polymarket_agent, run_agent
from polymarket_agents.graph.planning_agent import analyze_bet

agent = create_polymarket_agent(model="gpt-4o-mini", temperature=0.1)
print(run_agent(agent, "Find the best political market to trade"))

result = analyze_bet("Will ETH hit $5k by year end?")
print(result.get("recommendation"))
```

### 🧵 Workflows & pipelines
- `scripts/python/refresh_markets.py` refreshes `data/markets.db` (one-shot or continuous).
- `scripts/python/category_workflow.py` categorizes markets across domains.
- `scripts/python/politics_workflow.py` ingests politics-tagged markets into `data/memory.db`.
- `scripts/python/data_pipeline.py` runs refresh + info management + monitoring.
- `scripts/python/sports_explorer.py` explores leagues, teams, and market types.

Examples:
```bash
python scripts/python/refresh_markets.py --continuous --interval 300
python scripts/python/category_workflow.py --category crypto
python scripts/python/politics_workflow.py --limit 50 --view
python scripts/python/data_pipeline.py --continuous --interval 600
python scripts/python/sports_explorer.py --leagues
```

### 📚 RAG (ChromaDB)
Build and query a local vector index:
```bash
python scripts/python/cli.py create-local-markets-rag ./data
python scripts/python/cli.py query-local-markets-rag ./data "Which markets mention rate cuts?"
```

### 🛰️ CLOB trading tools
- Read-only and trading tools live in `src/polymarket_agents/langchain/clob_tools.py`.
- Simple trade helpers live in `src/polymarket_agents/tools/trade_tools.py`.
- Trading requires a wallet private key and (optionally) CLOB API credentials.

### 🖥️ Server
A minimal FastAPI server lives at `scripts/python/server.py`.
```bash
python scripts/python/server.py
```

### 🐳 Docker
```bash
./scripts/bash/build-docker.sh
./scripts/bash/run-docker.sh
./scripts/bash/run-docker-dev.sh
```

### ⏱️ Cron automation
Schedule pipelines and backups:
```bash
./scripts/bash/setup_cron_jobs.sh
```

### ⚡ Quick fetch example
```bash
python examples/fetch_active_bets.py
```

## 🧰 Scripts index (all entry points)

### 🐍 Python
- `scripts/python/cli.py` CLI for markets, agents, and tools
- `scripts/python/refresh_markets.py` refresh local markets DB
- `scripts/python/category_workflow.py` multi-category ingestion
- `scripts/python/politics_workflow.py` politics-only workflow
- `scripts/python/data_pipeline.py` full pipeline + monitoring
- `scripts/python/sports_explorer.py` sports metadata explorer
- `scripts/python/server.py` FastAPI server
- `scripts/python/setup.py` loads `.env` for dev helpers

### 🐚 Bash / Shell
- `scripts/bet_planner.sh` planning agent + portfolio tools
- `scripts/polymarket_agent.sh` memory agent shortcut
- `scripts/bash/install.sh` pip install helper
- `scripts/bash/build-docker.sh` build container
- `scripts/bash/run-docker.sh` run container
- `scripts/bash/run-docker-dev.sh` run container with bind mount
- `scripts/bash/start-dev.sh` FastAPI dev helper
- `scripts/bash/setup_cron_jobs.sh` cron + backup setup

### 🧪 Test & validation scripts
- `scripts/run_graph_tests.py` run graph unit/perf/e2e tests
- `scripts/validate_graphs.py` validate graph compilation
- `scripts/test_graph.py` small graph smoke test
- `scripts/test_trade_tools.py` mock trade tool tests
- `examples/fetch_active_bets.py` lightweight Gamma API sample

## 🧱 Data & storage
- `data/markets.db` main SQLite store (markets, news, price_history, bets, research, analytics).
- `data/memory.db` ingestion/politics workflows (same schema, different dataset).
- `data/ingested_markets_*.json` raw ingestion snapshots.
- `logs/refresh_daemon.log` + pipeline logs.
- `.langgraph_api/` local LangGraph runtime artifacts.

## 🔌 Integrations
- Polymarket Gamma API (market discovery)
- Polymarket CLOB API (orderbook + trading)
- LangChain + LangGraph
- OpenAI (LLM + embeddings)
- NewsAPI + Tavily (research)
- ChromaDB (RAG)
- FastAPI (server)

## 🧪 Tests & validation
Run all tests:
```bash
python -m pytest
python -m unittest
```

Graph checks:
```bash
python scripts/validate_graphs.py
python scripts/run_graph_tests.py
```

Tests include LangChain tools, LangGraph nodes, CLOB tools, E2E graph tests, and perf checks.

## 🧯 Troubleshooting
- `OPENAI_API_KEY` missing: set it for LLM + embeddings.
- `POLYGON_WALLET_PRIVATE_KEY` missing: required for trading tools.
- `TAVILY_API_KEY` or `NEWSAPI_API_KEY` missing: required only for those connectors.
- Empty results or slow agents: seed `data/markets.db` via `refresh_markets.py`.
- `py_clob_client` import errors: reinstall dependencies from `requirements.txt`.

## 🛡️ Safety & risk
- Trading is real money. Use small sizes and sanity-check recommendations.
- Keep private keys local and never commit `.env`.
- Consider running read-only workflows when prototyping.

## 📚 Docs
- `docs/LANGCHAIN_REFERENCE.md` LangChain integration + tools
- `docs/POLYMARKET_API_REFERENCE.md` Gamma/CLOB/Data/WebSocket reference
- `docs/WORKFLOW_POLITICS.md` politics pipeline diagram + schema
- `docs/EXAMPLE.md` sample run output

## 🙌 Contributing & license
- `CONTRIBUTING.md` covers contribution guidelines.
- `LICENSE.md` contains the license.

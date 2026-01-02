# Polymarket LangChain Agent

This project recreates the core components from the Polymarket Agents repository and integrates them with LangChain and LangGraph so you can build AI-powered research and trading workflows on Polymarket.

## Highlights
- Polymarket API clients (Gamma + CLOB) with trading support
- LangChain tools for markets, events, research, and orders
- LangGraph agents for memory-first analysis and statistical planning
- Local SQLite storage and categorized market databases
- RAG via ChromaDB, plus news and web search connectors
- CLI, workflow scripts, Docker helpers, and tests

## Repository layout
- `agents/` core library
  - `agents/polymarket/` Polymarket API clients (`polymarket.py`, `gamma.py`)
  - `agents/langchain/` LangChain tools and agent helpers (`tools.py`, `clob_tools.py`, `agent.py`)
  - `agents/graph/` LangGraph agents and state (`memory_agent.py`, `planning_agent.py`, `state.py`)
  - `agents/application/` app workflows (`trade.py`, `executor.py`, `creator.py`, `cron.py`, `prompts.py`)
  - `agents/tools/` tool wrappers (`market_tools.py`, `trade_tools.py`, `research_tools.py`)
  - `agents/connectors/` external data and RAG (`chroma.py`, `news.py`, `search.py`)
  - `agents/memory/` SQLite memory manager (`manager.py`)
  - `agents/team/` ingestion team orchestration (`ingestion.py`)
  - `agents/utils/` Pydantic models and helpers (`objects.py`, `utils.py`)
  - `agents/tooling.py` shared tool wrapper utilities
- `scripts/` runnable entry points and utilities
  - `scripts/python/` workflows and services (`cli.py`, `data_pipeline.py`, `refresh_markets.py`, `category_workflow.py`, `politics_workflow.py`, `sports_explorer.py`, `run_ingestion_team.py`, `server.py`, `setup.py`)
  - `scripts/bash/` Docker helpers and cron setup
  - `scripts/` root scripts (`bet_planner.sh`, `polymarket_agent.sh`, `run_graph_tests.py`, `validate_graphs.py`, `test_graph.py`, `test_trade_tools.py`)
- `docs/` reference docs and examples
- `tests/` pytest/unittest suites for tools, graphs, and integrations
- `data/` SQLite databases and ingested market snapshots
- `logs/` runtime logs
- `langgraph.json` LangGraph project configuration
- `fetch_active_bets.py` quick sample market fetch
- `Dockerfile`, `environment.yml`, `requirements.txt`, `package-lock.json`
- `.langgraph_api/` local LangGraph runtime artifacts
- `CONTRIBUTING.md`, `LICENSE.md`

## Quick start

1) Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate polymarket-agent
```

2) Or install with pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional helper script:
```bash
./scripts/bash/install.sh
```

3) Configure environment variables:
```bash
cp .env.example .env
```

## Environment variables

Required for LLM-powered agents:
- `OPENAI_API_KEY`

Required for trading (CLOB):
- `POLYGON_WALLET_PRIVATE_KEY` (or `PK`)

Optional connectors and tooling:
- `NEWSAPI_API_KEY` (NewsAPI)
- `TAVILY_API_KEY` (Tavily web search)
- `ANTHROPIC_API_KEY` (deep research agent, if available)
- `CLOB_API_URL` (default: https://clob.polymarket.com)
- `CHAIN_ID` (default: 137 / Polygon)
- `CLOB_API_KEY`, `CLOB_SECRET`, `CLOB_PASS_PHRASE` (use if you do not want to derive creds)

## Usage

### CLI
The main CLI lives at `scripts/python/cli.py` (Typer). Run:
```bash
python scripts/python/cli.py --help
```

Common commands:
- `get-all-markets` - list tradeable markets
- `get-all-events` - list events
- `get-relevant-news` - query NewsAPI
- `create-local-markets-rag` - build a local Chroma RAG index
- `query-local-markets-rag` - query the local RAG index
- `ask-superforecaster` - forecast a specific market
- `create-market` - draft a market description
- `ask-llm` - ask the base LLM
- `ask-polymarket-llm` - ask LLM with live markets/events context
- `run-autonomous-trader` - run the trading workflow
- `run-memory-agent` - LangGraph memory agent
- `run-planning-agent` - LangGraph planning agent
- `scan-opportunities` - find value opportunities
- `list-agents` - list orchestrator agents
- `run-deep-research-agent` - deep research agent (requires `agents/deep_research_agent.py` and ANTHROPIC/TAVILY keys)

Examples:
```bash
python scripts/python/cli.py get-all-markets --limit 5
python scripts/python/cli.py get-relevant-news "market keywords"
python scripts/python/cli.py run-memory-agent "Find interesting political markets"
python scripts/python/cli.py run-planning-agent "Will BTC hit 100k?"
```

### LangGraph config
The LangGraph project configuration lives in `langgraph.json` and wires:
- `memory_agent` to `agents/graph/memory_agent.py:create_memory_agent`
- `planning_agent` to `agents/graph/planning_agent.py:create_planning_agent`

### Agent entry points
- `scripts/polymarket_agent.sh` - run the memory agent via CLI.
- `scripts/bet_planner.sh` - run the planning agent, scan for opportunities, or show portfolio summary.

Examples:
```bash
./scripts/polymarket_agent.sh "Find crypto arbitrage opportunities"
./scripts/bet_planner.sh "Bitcoin 100k"
./scripts/bet_planner.sh --scan politics
./scripts/bet_planner.sh --portfolio
```

### Workflows and data pipelines
- `scripts/python/refresh_markets.py` - refresh the local markets database (one-shot or continuous).
- `scripts/python/data_pipeline.py` - full data pipeline and information management.
- `scripts/python/politics_workflow.py` - politics-only ingestion workflow.
- `scripts/python/category_workflow.py` - multi-category ingestion and stats.
- `scripts/python/run_ingestion_team.py` - orchestrated ingestion team.
- `scripts/python/sports_explorer.py` - explore sports leagues, teams, and market types.

Examples:
```bash
python scripts/python/refresh_markets.py --continuous --interval 300
python scripts/python/data_pipeline.py --continuous --interval 600
python scripts/python/politics_workflow.py --limit 50
python scripts/python/category_workflow.py --category crypto
python scripts/python/sports_explorer.py --leagues
```

### RAG and local memory
- `agents/connectors/chroma.py` implements ChromaDB-based RAG.
- Local SQLite data lives in `data/markets.db` and `data/memory.db`.

### Server
A minimal FastAPI server lives at `scripts/python/server.py`.

```bash
python scripts/python/server.py
```

For local dev with the FastAPI CLI:
```bash
./scripts/bash/start-dev.sh
```

### Docker
```bash
./scripts/bash/build-docker.sh
./scripts/bash/run-docker.sh
./scripts/bash/run-docker-dev.sh
```

### Cron automation
Set up scheduled pipelines and backups:
```bash
./scripts/bash/setup_cron_jobs.sh
```

### Quick fetch example
```bash
python fetch_active_bets.py
```

## Tests and validation

Pytest or unittest:
```bash
python -m pytest
python -m unittest
```

Graph-specific checks:
```bash
python scripts/validate_graphs.py
python scripts/run_graph_tests.py
python scripts/test_graph.py
```

Trade tool mocks:
```bash
python scripts/test_trade_tools.py
```

## Docs

- `docs/LANGCHAIN_REFERENCE.md` - LangChain integration, tool reference, best practices
- `docs/POLYMARKET_API_REFERENCE.md` - Gamma, CLOB, data APIs, and WebSocket reference
- `docs/WORKFLOW_POLITICS.md` - politics workflow diagram and schema
- `docs/EXAMPLE.md` - sample run output

## Contributing and license

- `CONTRIBUTING.md` covers contribution guidelines.
- `LICENSE.md` contains the license.

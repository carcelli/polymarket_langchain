# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a LangChain/LangGraph-based research and trading system for Polymarket prediction markets. It uses SQLite for local caching, integrates with Polymarket's Gamma API (market discovery) and CLOB API (trading), and provides two main LangGraph agents for analysis.

**Core Package**: `polymarket_agents` (located in `src/polymarket_agents/`)

## Essential Setup Commands

### Environment Setup
```bash
# Recommended: Use conda
conda env create -f environment.yml
conda activate polymarket-agent

# Alternative: Use pip
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure environment
cp .env.example .env
# Edit .env to add required API keys (OPENAI_API_KEY, POLYGON_WALLET_PRIVATE_KEY)

# Set PYTHONPATH for running modules directly
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
```

### Database Initialization
```bash
# Seed local database with market data (recommended before first use)
python scripts/python/refresh_markets.py --max-events 200

# Continuous refresh (for development/monitoring)
python scripts/python/refresh_markets.py --continuous --interval 300
```

### Running Tests
```bash
# Run all tests
python -m pytest
python -m unittest

# Graph validation tests
python scripts/validate_graphs.py
python scripts/run_graph_tests.py

# Test specific modules
python -m pytest tests/test_tools.py
python -m pytest tests/graph/
```

### Development Commands
```bash
# Run CLI (see all commands)
python scripts/python/cli.py --help

# Run agents directly
python scripts/python/cli.py run-memory-agent "Find interesting political markets"
python scripts/python/cli.py run-planning-agent "Will BTC hit 100k by year end?"

# Run agents as modules
python -m polymarket_agents.graph.memory_agent "What are the top crypto markets?"
python -m polymarket_agents.graph.planning_agent "Will the Fed cut rates in Q1 2025?"

# Run workflows
python scripts/python/category_workflow.py --category crypto
python scripts/python/politics_workflow.py --limit 50 --view
python scripts/python/data_pipeline.py --continuous --interval 600

# Start FastAPI server
python scripts/python/server.py
```

### Docker Commands
```bash
./scripts/bash/build-docker.sh
./scripts/bash/run-docker.sh          # Production mode
./scripts/bash/run-docker-dev.sh      # Development mode with bind mount
```

## Architecture Overview

### Core Design Patterns

#### 1. **Database-First, API-Second Pattern**
The system prioritizes local SQLite queries over API calls to reduce latency and costs:
- Query local `data/markets.db` first (instant, 20k+ markets cached)
- Enrich with live Gamma API data only when needed (explicit "live" requests, empty results, stale cache)
- Memory Agent implements this pattern in its memory → enrichment → reasoning flow

#### 2. **Hybrid ML + LLM Architecture**
The Planning Agent attempts XGBoost model predictions first, then falls back to GPT-4o-mini:
```python
try:
    probability = xgboost_model.predict()  # Fast, trained model
except:
    probability = llm.estimate_probability()  # Fallback: LLM reasoning
```

#### 3. **LangGraph State Management**
All agents use `AgentState` (TypedDict) for state flow:
```python
class AgentState(TypedDict):
    messages: List[BaseMessage]
    markets: List[Market]
    forecast_prob: float
    trade_decision: dict
    error: Optional[str]
```

#### 4. **Tool Composition over Specialization**
Rather than 100 specialized tools, the codebase provides ~30 composable primitives:
- Market discovery: `fetch_all_markets`, `get_current_markets_gamma`
- Database queries: `search_markets_db`, `get_markets_by_category`
- Analysis: `get_superforecast`, `analyze_market_with_llm`
- Trading: `preview_order`, CLOB tools (when wallet configured)

Tools are grouped by capability: `get_read_only_tools()`, `get_analysis_tools()`, `get_database_tools()`

### Two Primary Agents

#### Memory Agent (`graph/memory_agent.py`)
**Purpose**: Fast, local-first market discovery and analysis

**Node Flow**: Memory → Enrichment → Reasoning → Decision

**Key Characteristics**:
- Queries local SQLite database before hitting APIs
- Smart API call optimization (only fetches live data when needed)
- Configurable category focus via `MARKET_FOCUS` env var
- Best for: Quick queries, batch analysis, cost-sensitive workflows

**Direct invocation**:
```bash
python -m polymarket_agents.graph.memory_agent "Find high-volume crypto markets"
```

#### Planning Agent (`graph/planning_agent.py`)
**Purpose**: Quantitative betting analysis with edge calculation

**Node Flow**: Research → Stats → Probability → Decision

**Key Characteristics**:
- Calculates edge: `estimated_prob - implied_prob`
- Computes Expected Value (EV) and Kelly criterion sizing
- Hybrid ML/LLM probability estimation
- Outputs structured recommendations: BET / WATCH / PASS
- Best for: Individual bet analysis, portfolio optimization

**Direct invocation**:
```bash
python -m polymarket_agents.graph.planning_agent "Will ETH hit $5k by year end?"
```

### Database Schema

**Primary Database**: `data/markets.db` (SQLite)

**Key Tables**:
- `markets`: Core market inventory (20k+ rows, indexed on category, volume, active status)
- `price_history`: Time-series data for ML features (indexed on market_id, timestamp)
- `bets`: Position tracking with PnL calculations
- `market_analytics`: Pre-computed edge, EV, Kelly fractions
- `research`: Cached news/analysis with sentiment scores

**MemoryManager** (`memory/manager.py`): Provides query interface without ORM overhead
- Methods: `get_market()`, `list_by_category()`, `search_markets()`, `list_top_volume_markets()`
- Uses raw SQL for performance on read-heavy workloads

### Tool Architecture (`langchain/tools.py`)

**Design Pattern**:
```python
def _tool_implementation(params) -> result:
    """Business logic"""
    pass

def wrap_tool(impl_fn) -> LangChain_Tool:
    """Converts implementation to LangChain tool"""
    pass
```

**Tool Registry**: `_TOOL_FUNCTIONS` dict maps names → implementation functions

**Lazy Initialization**: Expensive objects (Gamma client, ChromaDB, News API) are initialized on first use via getter functions (`_get_gamma()`, `_get_memory()`, `_get_chroma()`)

### Configuration (`config.py`)

**Critical Environment Variables**:
- `OPENAI_API_KEY`: Required for LLM/embeddings
- `POLYGON_WALLET_PRIVATE_KEY` or `POLYGON_WALLET_KEY_FILE`: Required for trading
- `MARKET_FOCUS`: Optional category filter (e.g., "sports", "politics", "crypto")
- `DEFAULT_MODEL`: LLM model selection (default: "gpt-4o-mini")
- `DATABASE_PATH`: SQLite database location (default: "data/markets.db")

**Wallet Key Loading**: Supports raw private key string OR PEM file path (safer)

**Model Configuration**:
- Analysis tasks: temperature 0.1 (factual)
- Creative tasks: temperature 0.3 (research)
- Structured output: temperature 0.0 (deterministic)

## Key Integrations

### Polymarket APIs
- **Gamma API** (`connectors/gamma.py`): Market/event discovery, read-only
- **CLOB API** (`connectors/polymarket.py`): Trading execution, requires wallet

### LangChain/LangGraph
- LangGraph config: `langgraph.json` (defines agent entry points)
- Registered agents: `memory_agent`, `planning_agent`
- All nodes are traced via LangSmith decorators for observability

### External Services
- **NewsAPI** (`connectors/news.py`): News enrichment for research
- **Tavily** (via tools): Web search for deep research
- **ChromaDB** (`connectors/chroma.py`): Vector search/RAG for semantic market queries

## Project Structure

```
src/polymarket_agents/
├── graph/               # LangGraph agents (memory_agent.py, planning_agent.py, state.py)
├── langchain/           # LangChain tools, agent wrappers (tools.py, clob_tools.py, agent.py)
├── connectors/          # API clients (gamma.py, polymarket.py, news.py, chroma.py)
├── application/         # High-level workflows (executor.py, prompts.py, cron.py)
├── tools/               # Tool wrappers (market_tools.py, trade_tools.py, research_tools.py)
├── memory/              # Database interface (manager.py)
├── domain/              # Domain models (models.py)
├── utils/               # Utilities (database.py, objects.py, config.py)
├── ml_foundations/      # ML utilities (nn.py, utils.py)
├── subagents/           # Specialized sub-agents (github_agent.py, risk_analysis.py, etc.)
└── automl/              # AutoML pipeline

scripts/
├── python/              # Python scripts (cli.py, refresh_markets.py, workflows)
└── bash/                # Shell scripts (Docker, installation, cron setup)
```

## Development Guidelines

### When Adding New Features

**New Data Source**:
1. Add connector in `connectors/`
2. Expose via tool in `langchain/tools.py`
3. Add to appropriate tool group (`get_analysis_tools()`, etc.)

**New Agent**:
1. Define in `graph/` with node functions
2. Register in `langgraph.json`
3. Use `AgentState` TypedDict for state management
4. Add tests in `tests/graph/`

**New ML Model**:
1. Train model and save to `data/models/`
2. Integrate in Planning Agent probability node with try/except fallback
3. Test with and without model file present

**New Market Category**:
1. Update `MARKET_FOCUS` validation in `config.py`
2. Ensure database indexes support category filtering
3. Add category-specific workflow if needed (see `politics_workflow.py`)

### Code Patterns to Follow

**Pydantic for Validation**: Use Pydantic models at API boundaries (`utils/objects.py`)

**TypedDict for State**: Use TypedDict for LangGraph state (not Pydantic, for performance)

**Lazy API Calls**: Always check local database before hitting external APIs

**Graceful Degradation**: Use try/except for optional features (ML models, external APIs)

**No ORM**: Direct SQLite queries for performance (see `memory/manager.py`)

**LangSmith Tracing**: All agent nodes should be decorated for observability

### Testing Approach

**Unit Tests**: Test individual tools and connectors in isolation
```bash
python -m pytest tests/test_tools.py -v
```

**Graph Tests**: Validate agent compilation and node execution
```bash
python scripts/run_graph_tests.py
```

**Integration Tests**: Test end-to-end workflows with mock APIs where possible

**Manual Testing**: Use CLI commands for smoke testing
```bash
python scripts/python/cli.py get-all-markets --limit 5
```

## Common Workflows

### Market Research Workflow
1. Seed database: `python scripts/python/refresh_markets.py`
2. Query markets: `python scripts/python/cli.py run-memory-agent "Find crypto markets"`
3. Analyze specific market: `python scripts/python/cli.py run-planning-agent "Will BTC hit 100k?"`

### Trading Workflow (Requires Wallet)
1. Configure `POLYGON_WALLET_PRIVATE_KEY` in `.env`
2. Check balance: `python scripts/python/cli.py` (use appropriate wallet tool)
3. Preview order: Use `preview_order` tool via CLI
4. Execute: Use CLOB trading tools (exercise caution, real money)

### RAG Workflow
1. Build vector index: `python scripts/python/cli.py create-local-markets-rag ./data`
2. Query semantically: `python scripts/python/cli.py query-local-markets-rag ./data "Which markets mention rate cuts?"`

### Continuous Monitoring
1. Set up cron: `./scripts/bash/setup_cron_jobs.sh`
2. Run data pipeline: `python scripts/python/data_pipeline.py --continuous --interval 600`

## Important Notes

### Safety and Risk Management
- Trading tools interact with real money on Polygon network
- Private keys are sensitive - never commit `.env` files
- Use small position sizes when testing
- Preview all orders before execution
- Consider running in read-only mode (omit `POLYGON_WALLET_PRIVATE_KEY`) during development

### Performance Considerations
- Local database queries are ~100x faster than API calls
- Memory Agent can handle 20k+ markets without pagination
- Use `--limit` flags in CLI commands to reduce token usage
- Set `MARKET_FOCUS` to constrain agent scope and reduce noise

### Debugging
- Set `DEBUG_MODE=true` in `.env` for verbose logging
- Check `logs/` directory for daemon logs
- Use LangSmith tracing for agent execution visualization
- Query `data/markets.db` directly with SQLite CLI for data inspection

## LangGraph Configuration

The `langgraph.json` file defines agent entry points:
```json
{
  "graphs": {
    "memory_agent": "./src/polymarket_agents/graph/memory_agent.py:create_memory_agent",
    "planning_agent": "./src/polymarket_agents/graph/planning_agent.py:create_planning_agent"
  }
}
```

Both agents can be invoked via:
- CLI: `python scripts/python/cli.py run-memory-agent "query"`
- Module: `python -m polymarket_agents.graph.memory_agent "query"`
- LangGraph Server: Deploy with `langgraph` CLI (see LangGraph docs)

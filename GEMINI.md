# Polymarket LangChain Agent

## Project Overview
This project builds autonomous AI agents for researching, forecasting, and trading on Polymarket prediction markets. It leverages **LangChain** and **LangGraph** to orchestrate workflows that combine local data analysis, external news enrichment, and quantitative modeling (ML + LLM) to identify trading opportunities.

**Key Architecture:**
- **Local-First Data:** Prioritizes querying a local SQLite database (`data/markets.db`) over live API calls for speed and cost efficiency.
- **Hybrid Intelligence:** Combines XGBoost models for quantitative probability estimation with LLMs (GPT-4o) for qualitative reasoning.
- **Agentic Workflow:** Uses LangGraph to manage state across complex reasoning chains (Memory Agent for discovery, Planning Agent for deep analysis).

## Environment Setup

### Prerequisites
- Python 3.10+
- Conda (recommended) or venv
- API Keys: OpenAI, and optionally Polygon Private Key (for trading), NewsAPI, Tavily.

### Installation
**Using Conda:**
```bash
conda env create -f environment.yml
conda activate polymarket-agent
```

**Using pip:**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configuration
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` to add your keys:
   - `OPENAI_API_KEY`: Essential for agents.
   - `POLYGON_WALLET_PRIVATE_KEY`: Required for placing trades (keep secure!).
   - `MARKET_FOCUS`: Optional (e.g., "politics", "crypto").

## Key Commands

### 1. Data Management
**Refresh Market Data:**
Essential first step to populate the local database.
```bash
# Seed the database (fetch 200 events)
python scripts/python/refresh_markets.py --max-events 200

# Continuous monitoring
python scripts/python/refresh_markets.py --continuous --interval 300
```

### 2. Running Agents
**Memory Agent (Fast Discovery):**
Quickly finds markets using local SQL queries.
```bash
python scripts/python/cli.py run-memory-agent "Find high volume crypto markets"
# OR via module
python -m polymarket_agents.graph.memory_agent "What are the top politics markets?"
```

**Planning Agent (Deep Analysis):**
Performs detailed analysis, edge calculation, and EV estimation.
```bash
python scripts/python/cli.py run-planning-agent "Will Bitcoin hit 100k by year end?"
```

### 3. Trading & Analysis
**CLI Interface:**
```bash
python scripts/python/cli.py --help
python scripts/python/cli.py get-all-markets --limit 5
python scripts/python/cli.py get-relevant-news "election"
```

**Paper Trading (Recommended):**
Prove edge before risking real funds.
```bash
python scripts/auto_paper_trader.py
```

## Project Structure

### Source Code (`src/polymarket_agents/`)
- **`graph/`**: Core LangGraph agent definitions (`memory_agent.py`, `planning_agent.py`) and state management (`state.py`).
- **`langchain/`**: LangChain tools and wrappers (`tools.py`, `clob_tools.py`).
- **`connectors/`**: API clients for Polymarket (Gamma, CLOB), NewsAPI, etc.
- **`memory/`**: SQLite database interface (`manager.py`).
- **`ml_foundations/`**: Machine learning utilities.
- **`application/`**: Higher-level application logic and prompts.

### Scripts (`scripts/`)
- **`python/`**: Python entry points for workflows, CLI, and servers.
- **`bash/`**: Shell scripts for Docker, installation, and cron jobs.

### Data (`data/`)
- **`markets.db`**: The central SQLite database storing markets, bets, and history.

## Development Conventions

- **Database-First:** Always check `data/markets.db` before making external API calls. The `MemoryManager` class is optimized for this.
- **State Management:** Use `AgentState` (TypedDict) for passing data between LangGraph nodes.
- **Validation:** Use Pydantic models (`utils/objects.py`) for clear data contracts at API boundaries.
- **Safety:**
    - Never commit `.env` or private keys.
    - Use "Paper Trading" scripts to validate strategies.
    - Set `POLYGON_WALLET_PRIVATE_KEY` only when ready to trade real funds.
- **Testing:**
    - Run unit tests: `python -m pytest`
    - Validate graphs: `python scripts/validate_graphs.py`

## Troubleshooting
- **Empty Results?** Run `refresh_markets.py` to seed the database.
- **Wallet Issues?** Verify `POLYGON_WALLET_PRIVATE_KEY` in `.env` matches your address.
- **Import Errors?** Ensure `src` is in your `PYTHONPATH`:
  ```bash
  export PYTHONPATH=$PYTHONPATH:$(pwd)/src
  ```

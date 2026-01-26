# Architecture

Polymarket prediction market agents with three integrated pillars: **Domain Agents**, **LangChain Tools**, and **ML Strategies**.

## System Overview

```
                          ┌─────────────────────────────────┐
                          │         LangChain Agent         │
                          │   (graph/memory_agent.py or     │
                          │    graph/planning_agent.py)     │
                          └───────────────┬─────────────────┘
                                          │
                          ┌───────────────▼─────────────────┐
                          │       LangChain Tools           │
                          │   (langchain/domain_tools.py)   │
                          └───────────────┬─────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
    ┌─────────▼─────────┐     ┌───────────▼───────────┐   ┌───────────▼───────────┐
    │  Domain Registry  │     │     ML Strategies     │   │   Market Research     │
    │ (domains/registry)│     │   (ml_strategies/)    │   │   (connectors/)       │
    └─────────┬─────────┘     └───────────────────────┘   └───────────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
┌───▼───┐          ┌────▼────┐
│Crypto │          │  NBA    │
│Domain │          │ Domain  │
└───┬───┘          └────┬────┘
    │                   │
    └─────────┬─────────┘
              │
    ┌─────────▼─────────┐
    │   Gamma/CLOB API  │
    │   (connectors/)   │
    └───────────────────┘
```

## Directory Structure

```
src/polymarket_agents/
├── domains/              # NEW: Domain-specific agents (plugin architecture)
│   ├── registry.py       # Domain registration and discovery
│   ├── base.py           # Base protocols
│   ├── crypto/           # Crypto binary price prediction
│   │   ├── agent.py      # CryptoAgent - scans and recommends
│   │   ├── scanner.py    # CryptoScanner - data collection
│   │   └── models.py     # Domain-specific types
│   └── nba/              # NBA games and player props
│       ├── agent.py      # NBAAgent with Log5 edge calculation
│       ├── scanner.py    # Game/prop scanner
│       └── models.py     # NBA-specific types
│
├── langchain/            # LangChain integration layer
│   ├── agent.py          # ReAct agent creation
│   ├── tools.py          # Market research tools
│   ├── domain_tools.py   # Bridge: domains → LangChain tools
│   └── clob_tools.py     # CLOB trading tools
│
├── graph/                # LangGraph orchestration
│   ├── memory_agent.py   # Conversational memory agent
│   ├── planning_agent.py # Hierarchical planning agent
│   └── state.py          # AgentState TypedDict
│
├── connectors/           # External API clients
│   ├── gamma.py          # Gamma Markets API
│   ├── polymarket.py     # Polymarket CLOB client
│   ├── chroma.py         # ChromaDB for RAG
│   └── news.py           # NewsAPI integration
│
├── ml_strategies/        # ML betting strategies
│   ├── base_strategy.py  # Abstract MLBettingStrategy
│   ├── registry.py       # Strategy registration (like domains)
│   ├── xgboost_strategy.py # Production XGBoost model
│   ├── neural_net_strategy.py # PyTorch neural network
│   ├── lstm_probability.py   # LSTM time-series
│   ├── knn_strategy.py   # K-nearest neighbors
│   ├── edge_detection.py # Edge calculation logic
│   └── market_prediction.py  # Market predictor
│
├── utils/                # Shared utilities
│   ├── database.py       # SQLite wrapper
│   ├── objects.py        # Core Pydantic models (Market, Event)
│   ├── config.py         # Configuration management
│   └── analytics.py      # Price trend analysis
│
├── context.py            # Dependency injection (AppContext)
└── config.py             # Application configuration
```

## Three Pillars

### 1. Domain Agents (NEW architecture)

Domain agents encapsulate market-specific logic. Each domain provides:
- **Scanner**: Fetches and filters markets from Gamma API
- **Agent**: Calculates edge and generates recommendations

```python
# Register a new domain
from polymarket_agents.domains import register_domain, DomainConfig

register_domain(DomainConfig(
    name="politics",
    description="Political prediction markets",
    agent_factory=lambda ctx: PoliticsAgent(),
    scanner_factory=lambda ctx: PoliticsScanner(),
    categories=["politics"],
))
```

**Available Domains:**
- `crypto` - Binary price prediction (BTC above $X)
- `nba` - Game outcomes and player props

### 2. LangChain Tools

Domain agents automatically become LangChain tools via `domain_tools.py`:

```python
from polymarket_agents.langchain.domain_tools import get_all_domain_tools

tools = get_all_domain_tools()
# Returns: [crypto_scan, crypto_scan_asset, nba_scan, nba_scan_games, ...]

# Use with any LangChain agent
from langchain.agents import create_react_agent
agent = create_react_agent(llm, tools=tools)
```

**Tool Categories:**
- **Domain tools** - `crypto_scan`, `nba_scan`, `nba_analyze_matchup`
- **Market research** - `market_get_history`, `search_markets`
- **Trading** - `place_order`, `cancel_order`, `get_positions`

### 3. ML Strategies

ML strategies predict market outcomes and calculate edge:

```python
from polymarket_agents.ml_strategies import (
    MLBettingStrategy,
    StrategyResult,
    MarketPredictor,
)
from polymarket_agents.ml_strategies.registry import (
    register_strategy,
    best_strategy,
)

# Use registry to find best strategy
result = best_strategy(market_data, min_edge=0.02)

# Or use specific strategy
from polymarket_agents.ml_strategies.xgboost_strategy import XGBoostStrategy
strategy = XGBoostStrategy()
strategy.train(training_data)
result = strategy.predict(market_data)
```

**Available Strategies:**
| Strategy | Type | Purpose |
|----------|------|---------|
| `xgboost_strategy` | XGBoost | Production-quality gradient boosting |
| `neural_net_strategy` | PyTorch | Deep learning for complex patterns |
| `lstm_probability` | LSTM | Time-series probability prediction |
| `knn_strategy` | K-NN | Distance-based market similarity |
| `simple_momentum` | Function | Lightweight price momentum |
| `edge_detection` | Logic | Edge calculation algorithms |

## Data Flow

```
1. Market Discovery
   Gamma API → Scanner → Filter by category/volume → Market list

2. Edge Calculation
   Market data → ML Strategy → StrategyResult(edge, confidence, recommendation)

3. Tool Execution
   LangChain Agent → Domain Tool → Agent.run() → Formatted recommendations

4. Trading (optional)
   Recommendation → CLOB Tools → Polymarket CLOB API → Order execution
```

## Entry Points

### Domain Agents (Recommended)
```bash
# Crypto markets
python -m polymarket_agents.domains.crypto.agent

# NBA markets
python -m polymarket_agents.domains.nba.agent --mode games
```

### LangGraph Agents
```bash
# Conversational agent with memory
python -m polymarket_agents.graph.memory_agent "Find crypto markets with edge"

# Planning agent with task decomposition
python -m polymarket_agents.graph.planning_agent "Analyze BTC 100k markets"
```

### CLI
```bash
# Main CLI
python scripts/python/cli.py

# ML pipeline
python scripts/python/ml_pipeline_cli.py
```

## Configuration

**Environment Variables:**
```bash
OPENAI_API_KEY=sk-...          # Required for LangChain agents
POLYGON_WALLET_PRIVATE_KEY=... # Optional: for trading
DATABASE_PATH=data/markets.db  # SQLite database path
```

**Dependency Injection:**
```python
from polymarket_agents.context import AppContext

# Create context with custom sources
context = AppContext(
    price_source=MyPriceAPI(),
    sports_source=MySportsAPI(),
    db_path="custom/path.db",
)

# Pass to domain agents
from polymarket_agents.domains.registry import get_domain
crypto = get_domain("crypto")
agent = crypto.create_agent(context)
```

## Adding a New Domain

1. Create domain directory:
```
src/polymarket_agents/domains/politics/
├── __init__.py
├── agent.py
├── scanner.py
└── models.py
```

2. Implement scanner and agent:
```python
# scanner.py
class PoliticsScanner:
    def scan(self) -> list[Market]:
        # Fetch markets from Gamma API
        # Filter by category="politics"
        return markets

# agent.py
class PoliticsAgent:
    def __init__(self, context=None):
        self.scanner = PoliticsScanner()

    def run(self) -> list[Recommendation]:
        markets = self.scanner.scan()
        # Calculate edge for each market
        return recommendations
```

3. Register in `domains/__init__.py`:
```python
from .registry import register_domain, DomainConfig
from .politics import PoliticsAgent, PoliticsScanner

register_domain(DomainConfig(
    name="politics",
    description="Political election and policy markets",
    agent_factory=lambda ctx: PoliticsAgent(context=ctx),
    scanner_factory=lambda ctx: PoliticsScanner(),
    categories=["politics"],
))
```

4. Tools are automatically available:
```python
from polymarket_agents.langchain.domain_tools import get_domain_tools
tools = get_domain_tools("politics")  # Returns [politics_scan]
```

## Testing

```bash
# Run all tests
python -m pytest

# Run specific test module
python -m pytest tests/test_langchain_tools.py -v

# Run with coverage
python -m pytest --cov=src/polymarket_agents
```

## Architecture Decisions

1. **Domain-centric design** - Each market category (crypto, nba, politics) is a self-contained domain with its own scanner, agent, and models.

2. **Plugin architecture** - Domains register themselves at import time. New domains are automatically discovered.

3. **Tool abstraction** - Domain agents are wrapped as LangChain tools, enabling use with any LangChain agent type.

4. **Strategy registry** - ML strategies register themselves and can be discovered/selected dynamically.

5. **Dependency injection** - External data sources (prices, sports data) are injected via context, enabling testing and flexibility.

# 🎯 Polymarket Agents - Essential Guide

> **Quick reference for developers** - Read this first!

---

## 📊 **Codebase Stats**

- **Total Lines**: 26,371 lines of Python
- **Modules**: 14 major subsystems
- **Files**: ~80 Python files
- **Complexity**: Production-grade ML forecasting system

---

## 🔥 **Top 10 Files You MUST Understand**

### 1. **`context.py`** (149 lines) ⭐⭐⭐⭐⭐
**The Foundation**: Dependency injection container
- **Why Critical**: Every module uses this for database, API clients, config
- **Pattern**: Protocol-based design with lazy singletons
- **First Thing to Learn**: How `get_context()` works

```python
from polymarket_agents.context import get_context

ctx = get_context()
db = ctx.get_memory_manager()      # SQLite database
gamma = ctx.get_gamma_client()      # Polymarket API
```

---

### 2. **`domains/registry.py`** (179 lines) ⭐⭐⭐⭐⭐
**The Plugin System**: How market types are registered
- **Why Critical**: Enables adding new domains (NFL, politics) without touching core
- **Pattern**: Factory pattern with auto-discovery
- **Key Function**: `register_domain()`, `get_domain()`

```python
from polymarket_agents.domains.registry import get_domain

crypto = get_domain("crypto")
agent = crypto.create_agent(context)
recommendations = agent.run()  # Get trade recommendations
```

---

### 3. **`memory/manager.py`** (1779 lines) ⭐⭐⭐⭐⭐
**The Database Layer**: All persistence operations
- **Why Critical**: Central source of truth for markets, bets, experiments
- **Schema**: 12 tables (markets, price_snapshots, bets, research_data, experiments, etc.)
- **Type Safety**: Full type annotations with `Optional` and `assert` guards

**Key Tables**:
```sql
markets          -- All Polymarket markets (live + historical)
price_snapshots  -- Time-series price data for ML training
bets             -- Placed bets with outcomes (P&L tracking)
research_data    -- News, analysis, model outputs
agent_executions -- LangGraph run logs (debugging)
experiments      -- AutoML experiments (hyperparams, metrics)
predictions      -- Model predictions for backtesting
```

---

### 4. **`langchain/tools.py`** (1492 lines) ⭐⭐⭐⭐
**The Tool Library**: 50+ LangChain tools for agents
- **Why Critical**: Agents need these to interact with the system
- **Pattern**: Function → Pydantic schema → `wrap_tool()`

**Tool Categories**:
- Market search: `get_top_volume_markets`, `search_markets_db`
- Price data: `get_price_history`, `analyze_price_trends`
- Trading: `get_orderbook`, `calculate_position_size`
- News: `search_news`, `get_sentiment_analysis`
- GitHub: `create_issue`, `analyze_repo`

---

### 5. **`automl/ml_tools.py`** (765 lines) ⭐⭐⭐⭐
**The ML Pipeline**: LangChain tools for AutoML
- **Why Critical**: Agents can train ML models autonomously
- **Tools**: `DataIngestionTool`, `ModelTrainingTool`, `PredictionTool`, `EvaluationTool`

```python
agent = create_agent_with_tools([AutoMLPipelineTool()])
result = agent.invoke("Train XGBoost on BTC markets from last year")
```

---

### 6. **`domains/crypto/scanner.py`** (~400 lines) ⭐⭐⭐⭐
**Crypto Market Scanner**: Finds profitable BTC/ETH markets
- **Why Critical**: Full example of domain-specific logic
- **Workflow**: Scan → Enrich → Filter → Find Edge → Size Position

**Key Methods**:
```python
scanner = CryptoScanner(price_source=MyPriceAPI())
result = scanner.scan()           # Fetch markets
enriched = scanner.enrich(result) # Add price data
tradeable = scanner.filter_tradeable(enriched, min_volume=5000)
edges = scanner.find_edge(tradeable)
```

---

### 7. **`core/async_utils.py`** (467 lines) ⭐⭐⭐⭐
**Async Task Management**: Production-ready coroutine patterns
- **Why Critical**: Long-running agents need robust exception handling
- **Based On**: Fluent Python Chapter 16 (coroutine state machines)

**Key Classes**:
- `TaskSupervisor`: Manages multiple async tasks with health monitoring
- `TaskMetrics`: Tracks restarts, errors, runtime
- `robust_task()`: Auto-restart on recoverable errors

**Use Case**: Background data collection daemons

---

### 8. **`connectors/polymarket.py`** (514 lines) ⭐⭐⭐⭐
**Trading Client**: CLOB order book integration
- **Why Critical**: Executes real trades on Polygon blockchain
- **Uses**: Web3.py for signing, py-clob-client for order submission

**Key Functions**:
```python
poly = Polymarket()
balance = poly.get_usdc_balance()              # Check wallet
orderbook = poly.get_orderbook(token_id)       # Get prices
order_id = poly.execute_order(                 # Place trade
    price=0.55, size=100, side="BUY", token_id=token_id
)
```

**⚠️ Security**: Requires `POLYGON_WALLET_PRIVATE_KEY` (signs real transactions!)

---

### 9. **`ml_foundations/nn.py`** (~300 lines) ⭐⭐⭐
**Neural Network from Scratch**: Pure NumPy implementation
- **Why Critical**: Educational + full control over architecture
- **Implements**: Forward/backward prop, activation functions, mini-batch training

```python
nn = NeuralNetwork(layer_sizes=[10, 20, 20, 1])
nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)
predictions = nn.predict(X_test)
```

---

### 10. **`application/trade.py`** (77 lines) ⭐⭐⭐
**Autonomous Trading Bot**: End-to-end execution
- **Why Critical**: Shows full system integration
- **Strategy**: `one_best_trade()` - evaluates all markets, picks best, executes

**Flow**:
```python
trader = Trader()
events = trader.polymarket.get_all_tradeable_events()
filtered = trader.agent.filter_events_with_rag(events)
markets = trader.agent.map_filtered_events_to_markets(filtered)
best_trade = trader.agent.source_best_trade(markets[0])
# trader.polymarket.execute_market_order(market, amount)  # Commented for safety
```

---

## 🏗️ **Architecture Hierarchy**

```
Application Layer
├─ trade.py                    # Autonomous trading bot
├─ executor.py                 # Agent execution orchestrator
└─ prompts.py                  # LLM prompt templates

Agent Layer (LangGraph)
├─ graph/memory_agent.py       # Multi-turn conversation agent
├─ graph/planning_agent.py     # ReAct planning agent
└─ graph/state.py              # Shared agent state

Tool Layer (LangChain)
├─ langchain/tools.py          # 50+ tools (search, trade, research)
├─ langchain/agent.py          # Pre-built agent factories
├─ langchain/domain_tools.py   # Auto-generated domain tools
└─ langchain/clob_tools.py     # Trading-specific tools

Domain Layer (Business Logic)
├─ domains/crypto/             # BTC/ETH price prediction
│   ├─ agent.py                # Orchestrator
│   ├─ scanner.py              # Market scanner + enrichment
│   ├─ models.py               # Data models
│   └─ data_collector.py       # Background data daemon
├─ domains/nba/                # NBA game outcomes + props
│   ├─ agent.py
│   ├─ scanner.py              # Log5 edge calculation
│   └─ models.py
└─ domains/registry.py         # Plugin system

ML Layer (Prediction Models)
├─ automl/                     # Automated ML pipeline
│   ├─ auto_ml_pipeline.py     # Orchestrator
│   ├─ data_ingestion.py       # Training data preparation
│   ├─ data_quality.py         # Validation + cleaning
│   ├─ ml_agent.py             # ML research agent
│   └─ ml_tools.py             # LangChain tools for ML
├─ ml_strategies/              # Individual models
│   ├─ market_prediction.py    # Ensemble predictor
│   ├─ xgboost_strategy.py     # Gradient boosting
│   ├─ lstm_probability.py     # Time series LSTM
│   ├─ knn_strategy.py         # K-nearest neighbors
│   ├─ edge_detection.py       # Find mispriced markets
│   └─ evaluation.py           # Metrics (Brier, log loss, Sharpe)
└─ ml_foundations/             # From-scratch implementations
    ├─ nn.py                   # NumPy neural network
    └─ utils.py                # Math utilities

Data Layer (External APIs)
├─ connectors/
│   ├─ gamma.py                # Polymarket Gamma API (market data)
│   ├─ polymarket.py           # CLOB trading client
│   ├─ news.py                 # NewsAPI integration
│   ├─ search.py               # Tavily web search
│   └─ chroma.py               # Vector database
└─ memory/
    └─ manager.py              # SQLite database (12 tables)

Infrastructure Layer
├─ context.py                  # Dependency injection
├─ config.py                   # Environment variables
├─ tooling.py                  # Tool wrapping abstraction
└─ core/
    └─ async_utils.py          # Async task supervision
```

---

## 🔑 **Key Design Patterns**

### 1. **Dependency Injection** (`context.py`)
**Problem**: Hardcoded paths break tests  
**Solution**: Injectable dependencies via protocols

```python
class PriceSource(Protocol):
    def get_current_price(self, asset: str) -> float: ...

@dataclass
class AppContext:
    price_source: Optional[PriceSource] = None
    
    def get_memory_manager(self) -> MemoryManager:
        if self._memory_manager is None:
            self._memory_manager = MemoryManager(self.db_path)
        return self._memory_manager
```

---

### 2. **Plugin Architecture** (`domains/registry.py`)
**Problem**: Adding new market types requires touching many files  
**Solution**: Domain registry with factory pattern

```python
register_domain(DomainConfig(
    name="crypto",
    description="Binary price prediction markets",
    agent_factory=lambda ctx: CryptoAgent(price_source=ctx.price_source),
    scanner_factory=lambda ctx: CryptoScanner(price_source=ctx.price_source),
    categories=["crypto"],
))
```

---

### 3. **Protocol-Based Design** (`domains/base.py`)
**Problem**: Concrete classes create tight coupling  
**Solution**: Protocols define contracts, implementations vary

```python
class EventScanner(ABC):
    @abstractmethod
    def scan(self) -> ScanResult: pass
    
    @abstractmethod
    def enrich(self, markets: list[Market]) -> list[Market]: pass
    
    @abstractmethod
    def filter_tradeable(self, markets, min_volume, min_liquidity) -> list[Market]: pass
```

---

### 4. **Tool Wrapping** (`tooling.py`)
**Problem**: LangChain dependency in every file  
**Solution**: Abstraction with graceful fallback

```python
def wrap_tool(func, *, name=None, description=None, args_schema=None):
    if LANGCHAIN_AVAILABLE:
        return StructuredTool.from_function(func, ...)
    return ToolWrapper(func, ...)  # Minimal fallback
```

---

### 5. **Separation of Concerns**
- **Connectors**: API clients (no business logic)
- **Domains**: Business logic (no API details)
- **Tools**: LangChain adapters (no ML code)
- **ML Strategies**: Prediction models (no API calls)

---

## 🚀 **Common Workflows**

### **Workflow 1: Analyze Crypto Markets**

```python
# 1. Setup context
from polymarket_agents.context import get_context, set_context, AppContext
ctx = AppContext(db_path="data/markets.db")
set_context(ctx)

# 2. Get domain
from polymarket_agents.domains.registry import get_domain
crypto_domain = get_domain("crypto")

# 3. Create agent
agent = crypto_domain.create_agent(ctx)

# 4. Run analysis
recommendations = agent.run()

# 5. Review results
for rec in recommendations:
    print(f"{rec.market.question}")
    print(f"  Edge: {rec.edge.edge:.1%}")
    print(f"  Action: {rec.action} at {rec.size_fraction:.1%} of bankroll")
```

---

### **Workflow 2: Train ML Model**

```python
# 1. Setup database
from polymarket_agents.memory.manager import MemoryManager
db = MemoryManager("data/markets.db")

# 2. Ingest data
from polymarket_agents.automl.data_ingestion import DataIngestion
ingestion = DataIngestion(db)
data = ingestion.fetch_training_data(days_back=365, min_volume=1000)

# 3. Train model
from polymarket_agents.ml_strategies.xgboost_strategy import XGBoostStrategy
model = XGBoostStrategy()
model.fit(data["X"], data["y"])

# 4. Generate predictions
predictions = model.predict_proba(data["X_test"])

# 5. Evaluate
from polymarket_agents.ml_strategies.evaluation import evaluate_model
metrics = evaluate_model(predictions, data["y_test"])
print(f"Brier Score: {metrics['brier_score']:.3f}")
print(f"Accuracy: {metrics['accuracy']:.1%}")
```

---

### **Workflow 3: Execute Trade**

```python
# 1. Initialize trading client
from polymarket_agents.connectors.polymarket import Polymarket
poly = Polymarket()  # Loads private key from env

# 2. Check balance
balance = poly.get_usdc_balance()
print(f"Balance: ${balance:.2f}")

# 3. Get market
token_id = "101669189743438912873361127612589311253202068943959811456820079057046819967115"
market = poly.get_market(token_id)

# 4. Check orderbook
orderbook = poly.get_orderbook(token_id)
best_bid = orderbook.bids[0].price
best_ask = orderbook.asks[0].price
print(f"Spread: {best_bid} - {best_ask}")

# 5. Place order
order_id = poly.execute_order(
    price=0.55,        # Limit price
    size=10,           # USDC amount
    side="BUY",        # BUY or SELL
    token_id=token_id
)
print(f"Order placed: {order_id}")
```

---

### **Workflow 4: Run LangChain Agent**

```python
# 1. Create agent
from polymarket_agents.langchain.agent import create_crypto_agent
agent = create_crypto_agent()

# 2. Invoke with query
result = agent.invoke({
    "input": "Find BTC markets with >5% edge and volume >$10k"
})

# 3. View result
print(result["output"])
```

---

## 📚 **Learning Path**

### **Week 1: Foundation**
- [ ] Read `config.py` (environment setup)
- [ ] Study `context.py` (dependency injection)
- [ ] Explore `connectors/gamma.py` (API client)
- [ ] Understand `domains/base.py` (core abstractions)
- [ ] Run `examples/` scripts

### **Week 2: Domains**
- [ ] Read `domains/registry.py` (plugin system)
- [ ] Study `domains/crypto/agent.py` (full pipeline)
- [ ] Explore `domains/crypto/scanner.py` (enrichment)
- [ ] Understand `domains/crypto/models.py` (data models)
- [ ] Implement a simple scanner

### **Week 3: LangChain**
- [ ] Read `langchain/tools.py` (tool patterns)
- [ ] Study `langchain/agent.py` (agent factories)
- [ ] Explore `graph/memory_agent.py` (LangGraph)
- [ ] Implement a new tool
- [ ] Build a custom agent

### **Week 4: ML**
- [ ] Read `automl/auto_ml_pipeline.py` (orchestration)
- [ ] Study `ml_strategies/market_prediction.py` (ensemble)
- [ ] Explore `ml_foundations/nn.py` (neural network)
- [ ] Train a model on historical data
- [ ] Evaluate performance metrics

---

## ⚠️ **Common Pitfalls**

### 1. **Hardcoding Paths**
❌ **Bad**: `db = MemoryManager("data/markets.db")`  
✅ **Good**: `db = get_context().get_memory_manager()`

### 2. **Ignoring Context**
❌ **Bad**: Creating new clients everywhere  
✅ **Good**: Use context singletons (`ctx.get_gamma_client()`)

### 3. **Forgetting Type Safety**
❌ **Bad**: `def get_markets() -> list:`  
✅ **Good**: `def get_markets() -> list[Market]:`

### 4. **Mixing Concerns**
❌ **Bad**: API calls in ML strategy  
✅ **Good**: Connectors → Domains → ML (separation)

### 5. **Not Using Protocols**
❌ **Bad**: Hardcoded to specific price API  
✅ **Good**: `PriceSource` protocol for testability

---

## 🎯 **Quick Wins**

### **Add a New Tool** (30 minutes)
```python
# 1. Define implementation
def _get_market_sentiment_impl(market_id: str) -> str:
    """Analyze market sentiment from news."""
    db = get_context().get_memory_manager()
    news = db.get_market_news(market_id)
    sentiment = analyze_sentiment(news)
    return json.dumps({"sentiment": sentiment})

# 2. Define schema
class MarketSentimentInput(BaseModel):
    market_id: str = Field(description="Market ID to analyze")

# 3. Wrap as tool
get_market_sentiment = wrap_tool(
    _get_market_sentiment_impl,
    name="get_market_sentiment",
    args_schema=MarketSentimentInput
)
```

### **Add a New Domain** (2-4 hours)
1. Create `domains/nfl/` directory
2. Define `NFLMarket(Market)` in `models.py`
3. Implement `NFLScanner(EventScanner)` in `scanner.py`
4. Write `NFLAgent` in `agent.py`
5. Register in `domains/registry.py`

### **Add a New ML Model** (4-6 hours)
1. Create `ml_strategies/random_forest.py`
2. Inherit from `BaseStrategy`
3. Implement `fit()` and `predict_proba()`
4. Register in `ml_strategies/registry.py`
5. Add to `AutoMLPipeline`

---

## 🔗 **File Dependencies**

```
context.py
├─ memory/manager.py (database)
├─ connectors/gamma.py (API)
└─ connectors/polymarket.py (trading)

domains/registry.py
├─ domains/crypto/agent.py
├─ domains/crypto/scanner.py
└─ domains/nba/agent.py

langchain/tools.py
├─ memory/manager.py (data queries)
├─ connectors/* (API calls)
└─ tooling.py (wrapping)

automl/auto_ml_pipeline.py
├─ automl/data_ingestion.py
├─ automl/data_quality.py
├─ ml_strategies/* (models)
└─ memory/manager.py (storage)
```

---

## 🎓 **Resources**

### **In-Codebase Documentation**
- `CLAUDE.md` - Setup + architecture overview
- `ARCHITECTURE.md` - Detailed system design
- `CODEBASE_BREAKDOWN.md` - Complete file-by-file breakdown (this was just created!)
- `examples/` - Runnable scripts

### **External References**
- LangChain docs: https://python.langchain.com/
- LangGraph docs: https://langchain-ai.github.io/langgraph/
- Polymarket API: https://docs.polymarket.com/
- py-clob-client: https://github.com/Polymarket/py-clob-client

---

## 🚦 **Status Check**

Run these to verify your setup:

```bash
# Check Python environment
python --version  # Should be 3.8+
pip list | grep -E "(langchain|polymarket|pandas|numpy)"

# Check database
python -c "from polymarket_agents.memory.manager import MemoryManager; db = MemoryManager('data/markets.db'); print(f'Markets: {len(db.get_all_markets())}')"

# Check API connectivity
python -c "from polymarket_agents.connectors.gamma import GammaMarketClient; gamma = GammaMarketClient(); print(f'Markets: {len(gamma.get_markets({\"limit\": 10}))}')"

# Check type safety
mypy src/polymarket_agents/context.py --show-error-codes
```

---

## 💡 **Pro Tips**

1. **Always use `get_context()`** - Never hardcode database paths
2. **Read tests first** - Best documentation of how code works
3. **Start with examples/** - Runnable code beats docs
4. **Use protocols for testability** - Mock external APIs easily
5. **Profile before optimizing** - Measure, don't guess
6. **Log everything** - Structured JSON logs for debugging
7. **Test with paper trading** - Validate before real money
8. **Monitor metrics** - Brier score, Sharpe ratio, win rate
9. **Version control experiments** - Use MLflow or Weights & Biases
10. **Document decisions** - ADRs (Architectural Decision Records)

---

## 📊 **Final Stats**

```
Total Lines of Code:  26,371
Total Files:          ~80
Modules:              14
Complexity:           HIGH (production-grade ML system)

Top 5 Largest Files:
1. memory/manager.py       1,779 lines (database layer)
2. langchain/tools.py      1,492 lines (tool library)
3. automl/ml_tools.py        765 lines (ML tools)
4. connectors/polymarket.py  514 lines (trading client)
5. core/async_utils.py       467 lines (async patterns)

Estimated Learning Time:
- Junior Dev:   4-6 weeks to proficiency
- Mid Dev:      2-3 weeks to proficiency
- Senior Dev:   1 week to proficiency
```

---

**Questions?** Dive into specific files or ask about patterns!


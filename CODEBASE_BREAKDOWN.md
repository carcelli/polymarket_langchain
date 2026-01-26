# 📚 Polymarket Agents Codebase - Complete Breakdown

> **Educational guide to every file in `src/polymarket_agents/`**
> Last Updated: 2026-01-26

---

## 🏗️ Architecture Overview

This project is a **multi-layered ML forecasting system** for Polymarket prediction markets, combining:
- **LangChain/LangGraph** for agent orchestration
- **Domain-specific agents** (crypto, NBA) with plugin architecture  
- **AutoML pipeline** for market prediction
- **Neural networks** (from scratch!) for probability estimation
- **Real-time data collection** from APIs (Polymarket, crypto prices, sports stats)

### **Design Philosophy**
- **Dependency Injection** via `context.py` - no hardcoded paths
- **Domain Registry** for pluggable agents (add new markets easily)
- **Separation of concerns**: connectors → domains → tools → agents
- **Type safety** with Pydantic and pandas-stubs for production reliability

---

## 📂 Module-by-Module Breakdown

---

## 1. 🔧 **Core Infrastructure**

### `__init__.py`
**Purpose**: Package marker (empty file)  
**Key Concept**: Makes `polymarket_agents` a Python package

### `config.py`
**Purpose**: Centralized configuration from environment variables  
**What It Does**:
- Loads API keys (OpenAI, Tavily, NewsAPI, Polygon wallet)
- CLOB trading config (Polymarket's order book)
- LLM model selection (`DEFAULT_MODEL`, temperature settings)
- Market focus filtering (sports/politics/crypto)
- Database path configuration

**Key Functions**:
```python
get_polygon_private_key()  # Loads wallet key from env or file
validate_config()           # Warns about missing required vars
```

**Pattern Used**: Environment-first configuration with sensible defaults
```python
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")  # Fallback to mini
MARKET_FOCUS = os.getenv("MARKET_FOCUS")  # Optional category filter
```

---

### `context.py`
**Purpose**: **Dependency Injection Container** (replaces singletons)  
**Key Concept**: Protocol-based design for testability

**Why This Exists**: Avoid hardcoded database paths and API clients scattered everywhere.

**Architecture**:
```python
@dataclass
class AppContext:
    db_path: str                           # Where SQLite lives
    price_source: Optional[PriceSource]    # Crypto price API (injectable)
    sports_source: Optional[SportsSource]  # NBA stats API (injectable)
    
    # Lazy-loaded singletons (cached after first access)
    def get_memory_manager(self) -> MemoryManager
    def get_gamma_client(self) -> GammaMarketClient
```

**Usage Pattern**:
```python
# At startup (once):
from polymarket_agents.context import set_context, AppContext
ctx = AppContext(db_path="test/test.db", price_source=MockPriceAPI())
set_context(ctx)

# Anywhere in code:
from polymarket_agents.context import get_context
db = get_context().get_memory_manager()
```

**Best Practice**: This is the **Single Source of Truth** for dependencies. Never hardcode `"data/markets.db"` in functions.

---

### `tooling.py`
**Purpose**: **Wrapper for LangChain tools** with graceful fallback  
**Key Concept**: Write tools once, use with or without LangChain

**Why This Exists**: Some environments (like pure Python scripts) don't need full LangChain. This provides a thin adapter layer.

**Class: `ToolWrapper`**
- Mimics LangChain's `StructuredTool` API
- Implements `.invoke()`, `.run()`, `__call__()` for compatibility

**Function: `wrap_tool()`**
```python
def wrap_tool(func, *, name=None, description=None, args_schema=None):
    if LANGCHAIN_AVAILABLE:
        return StructuredTool.from_function(func, ...)  # Full LangChain tool
    return ToolWrapper(func, ...)  # Minimal fallback
```

**Example**:
```python
def get_top_markets(limit: int = 10) -> str:
    """Fetch top volume markets."""
    return db.fetch_markets(limit)

# Wrap it (works with or without LangChain installed)
tool = wrap_tool(get_top_markets, name="get_top_markets")
result = tool.invoke({"limit": 5})
```

---

## 2. 🔌 **Connectors** (External APIs)

### `connectors/gamma.py`
**Purpose**: **Polymarket Gamma API client** (market data)  
**What It Does**: Fetches markets, events, tags from Polymarket's public API

**Key Class: `GammaMarketClient`**
```python
gamma = GammaMarketClient()
markets = gamma.get_markets({"active": True, "limit": 100})
events = gamma.get_events()
tags = gamma.get_all_tags()
```

**Important Methods**:
- `get_current_markets()` - Active, non-closed, non-archived only
- `get_all_current_markets()` - Paginated batch fetching (handles 1000s of markets)
- `get_clob_tradeable_markets()` - Markets with order book enabled
- `parse_pydantic_market()` - Converts JSON → Pydantic `Market` objects

**Data Flow**:
```
Gamma API (JSON) → httpx.get() → parse_pydantic_market() → Market(Pydantic)
```

**Why Pydantic?**: Type-safe data models catch API schema changes at parse time.

---

### `connectors/polymarket.py`
**Purpose**: **CLOB client for trading** (Polymarket's order book)  
**What It Does**: Places trades, approves tokens, fetches balances

**Key Class: `Polymarket`**
```python
poly = Polymarket()  # Auto-loads private key from env
balance = poly.get_usdc_balance()
orderbook = poly.get_orderbook(token_id)
order_id = poly.execute_order(price=0.55, size=10, side="BUY", token_id="...")
```

**Critical Concepts**:
1. **Web3 Integration**: Uses `web3.py` to sign transactions on Polygon blockchain
2. **Token Approvals**: USDC and CTF (Conditional Tokens) must be approved before trading
   - `_init_approvals()` - One-time setup (approves contracts to spend your tokens)
3. **Order Building**: Creates signed limit orders using `py_order_utils`

**Trading Flow**:
```python
# 1. Get market
market = poly.get_market(token_id)

# 2. Check orderbook
orderbook = poly.get_orderbook(token_id)
best_bid = orderbook.bids[0].price

# 3. Place order
order = poly.execute_order(
    price=0.52,        # Limit price
    size=100,          # USDC amount
    side="BUY",        # BUY or SELL
    token_id=token_id
)
```

**Security Note**: `POLYGON_WALLET_PRIVATE_KEY` must be kept secret. This signs real blockchain transactions!

---

### `connectors/news.py`
**Purpose**: **NewsAPI integration** for market research  
**What It Does**: Fetches news articles for market keywords

**Key Class: `News`**
```python
news = News()
articles = news.get_articles_for_options(["Trump", "Biden"])
```

**Use Case**: Enrich agent context with recent news for better probability estimates.

---

### `connectors/search.py`
**Purpose**: **Tavily search API** for web research  
**What It Does**: Search web for context (used by LangChain agents)

**Key Class: `Search`**
```python
search = Search()
context = search.search_context("Will BTC hit $100k in 2024?")
```

**Why Tavily?**: Optimized for LLM context (summarizes results, removes boilerplate).

---

### `connectors/chroma.py`
**Purpose**: **Vector database client** (not shown, but inferred from structure)  
**What It Does**: Stores embeddings for semantic search over historical market data

---

## 3. 🎯 **Domains** (Pluggable Market Types)

### **Architecture: Plugin System**

The domain registry enables **plug-and-play market types**. Want to add NFL markets? Just register a new domain!

### `domains/registry.py`
**Purpose**: **Central registry for domain plugins**  
**Key Concept**: Factory pattern + dependency injection

**Core Classes**:
```python
@dataclass
class DomainConfig:
    name: str                                    # "crypto", "nba"
    description: str                             # For tool docs
    agent_factory: Callable[[DataContext], Any]  # Creates agent
    scanner_factory: Callable[[DataContext], Any]  # Creates scanner
    categories: list[str]                        # Polymarket categories
    tags: list[str]                              # Metadata
```

**Registration**:
```python
from polymarket_agents.domains.registry import register_domain, DomainConfig

register_domain(DomainConfig(
    name="crypto",
    description="Binary price prediction markets for BTC, ETH",
    agent_factory=lambda ctx: CryptoAgent(price_source=ctx.price_source),
    scanner_factory=lambda ctx: CryptoScanner(price_source=ctx.price_source),
    categories=["crypto"],
    tags=["bitcoin", "ethereum", "price"]
))
```

**Discovery**:
```python
crypto = get_domain("crypto")
agent = crypto.create_agent(context)  # Injects dependencies
recs = agent.run()  # Get trade recommendations
```

**Auto-Registration**: `_register_builtin_domains()` runs on import, registering crypto and NBA.

---

### `domains/base.py`
**Purpose**: **Base protocols for all domains**  
**Key Concept**: Contract-based design (Protocol pattern)

**Core Classes**:

#### `Market` (Base dataclass)
```python
@dataclass
class Market:
    id: str
    question: str
    yes_price: float      # 0.0 to 1.0
    volume: float
    liquidity: float
    end_date: datetime
    token_id: str         # For trading
    
    @property
    def implied_prob(self) -> float:
        return self.yes_price  # Market's probability estimate
```

#### `EventScanner` (Protocol)
All domain scanners implement:
```python
class EventScanner(ABC):
    @abstractmethod
    def scan(self) -> ScanResult:
        """Fetch all relevant markets for this domain."""
        
    @abstractmethod
    def enrich(self, markets: list[Market]) -> list[Market]:
        """Add external data (prices, stats, news)."""
        
    @abstractmethod
    def filter_tradeable(self, markets, min_volume, min_liquidity) -> list[Market]:
        """Return markets worth trading."""
```

#### `Edge` (Calculated edge on a market)
```python
@dataclass
class Edge:
    market_id: str
    our_prob: float       # Our ML estimate
    market_prob: float    # Market's implied probability
    
    @property
    def edge(self) -> float:
        return self.our_prob - self.market_prob  # +ve = YES underpriced
    
    def kelly_fraction(self, bankroll: float) -> float:
        """Kelly criterion position sizing."""
        # f* = (bp - q) / b
```

**Why This Design?**: Each domain (crypto, NBA, politics) follows the same workflow:
1. **Scan** Polymarket for relevant markets
2. **Enrich** with external data (prices, stats)
3. **Filter** to tradeable markets (volume, liquidity)
4. **Find edge** (our prob vs market prob)
5. **Size position** (Kelly criterion)

---

### **Crypto Domain**

#### `domains/crypto/agent.py`
**Purpose**: **Orchestrator for crypto markets**  
**What It Does**: Runs full pipeline: scan → enrich → find edge → recommend

**Key Class: `CryptoAgent`**
```python
agent = CryptoAgent(
    price_source=MyPriceAPI(),
    min_volume=5000,      # Skip low-volume markets
    min_edge=0.05         # 5% edge minimum
)
recs = agent.run()        # Get top recommendations

for rec in recs:
    print(f"{rec.market.question}")
    print(f"  Edge: {rec.edge.edge:.1%}")
    print(f"  Action: {rec.action} at {rec.size_fraction:.1%} of bankroll")
```

**Output Example**:
```
Will BTC be above $100k on Dec 31?
  Edge: +12.3%
  Action: BUY_YES at 6.2% of bankroll
  Reasoning: Market at 45%, our model estimates 57%. Strong upward momentum.
```

#### `domains/crypto/scanner.py`
**Purpose**: **Fetches and enriches crypto markets**  
**What It Does**:
1. Scans Polymarket for "BTC", "ETH", "crypto" markets
2. Parses questions → extracts target price + date
3. Fetches historical prices from `price_source`
4. Calculates edge using probability models

**Key Methods**:
- `scan()` - Fetch all crypto markets from Gamma API
- `enrich()` - Add price history, volatility, momentum
- `filter_tradeable()` - Apply volume/liquidity/edge filters
- `find_edge()` - Calculate our prob vs market prob

#### `domains/crypto/models.py`
**Purpose**: **Data models for crypto domain**  
**What It Does**: Defines `CryptoPriceMarket`, `Asset` enum, `PriceDataSource` protocol

```python
@dataclass
class CryptoPriceMarket(Market):
    asset: Asset           # BTC, ETH, SOL
    target_price: float    # "Will BTC hit $100k?"
    direction: str         # "above" or "below"
    
    # Enriched data (added by scanner)
    current_price: Optional[float] = None
    price_history: Optional[list] = None
    volatility: Optional[float] = None
```

#### `domains/crypto/data_collector.py`
**Purpose**: **Continuous data collection daemon**  
**What It Does**: Runs as a background service, collecting price snapshots every N minutes

**Key Features**:
- Tracks all crypto markets in database
- Fetches live prices from `ccxt` (exchanges like Binance)
- Stores snapshots for training ML models
- Detects resolved markets (updates outcomes)

**Use Case**: Build training dataset for LSTM/XGBoost models.

---

### **NBA Domain**

#### `domains/nba/agent.py`
**Purpose**: **Orchestrator for NBA markets**  
**What It Does**: Same pipeline as crypto, but for basketball

**Key Difference**: Two market types:
1. **Game outcomes** (Team A to beat Team B)
2. **Player props** (LeBron over 25.5 points)

```python
agent = NBAAgent(data_source=MyNBAStatsAPI())
recs = agent.run()

# Or focus on one type
game_recs = agent.scan_games()
prop_recs = agent.scan_props()
```

#### `domains/nba/scanner.py`
**Purpose**: **Fetches and enriches NBA markets**  
**What It Does**:
1. Scans Polymarket for NBA game/prop markets
2. Parses questions → extracts teams/players/lines
3. Fetches team stats, injuries from `data_source`
4. Calculates edge using **Log5 method** (baseball sabermetrics)

**Log5 Formula**:
```python
def log5_prob(team_a_win_pct, team_b_win_pct) -> float:
    """Adjusted win probability accounting for opponent strength."""
    pa = team_a_win_pct
    pb = team_b_win_pct
    return (pa - pa * pb) / (pa + pb - 2 * pa * pb)
```

#### `domains/nba/models.py`
**Purpose**: **Data models for NBA domain**  
**What It Does**: Defines `NBAGameMarket`, `NBAPlayerProp`, `SportsDataSource` protocol

```python
@dataclass
class NBAGameMarket(Market):
    home_team: str
    away_team: str
    game_date: datetime
    
    # Enriched data
    home_win_pct: Optional[float] = None
    away_win_pct: Optional[float] = None
    injuries: Optional[list[str]] = None
    
@dataclass
class NBAPlayerProp(Market):
    player: str
    stat_type: str      # "points", "rebounds", "assists"
    line: float         # Over/under this value
    player_avg: Optional[float] = None
```

---

## 4. 🤖 **LangChain Integration** (Tools & Agents)

### `langchain/tools.py` ⚠️ **LARGE FILE (1492 lines)**
**Purpose**: **Bridge between Python functions → LangChain tools**  
**What It Does**: Wraps every database/API function as a LangChain tool

**Key Pattern**:
```python
# 1. Define implementation
def _get_top_volume_markets_impl(limit: int = 10, category: Optional[str] = None) -> str:
    """Fetch top markets by volume from database."""
    db = get_context().get_memory_manager()
    markets = db.get_top_markets(limit, category)
    return json.dumps(markets, indent=2)

# 2. Define Pydantic schema (for LangChain validation)
class TopVolumeMarketsInput(BaseModel):
    limit: int = Field(default=10, description="Max markets to return")
    category: Optional[str] = Field(None, description="Filter by category")

# 3. Wrap as tool
get_top_volume_markets = wrap_tool(
    _get_top_volume_markets_impl,
    name="get_top_volume_markets",
    args_schema=TopVolumeMarketsInput
)
```

**Available Tools** (50+ functions):
- **Market Search**: `search_markets_db`, `get_markets_by_category`, `get_top_volume_markets`
- **Price Data**: `get_price_history`, `analyze_price_trends`, `calculate_volatility`
- **Trading**: `get_orderbook`, `calculate_position_size`, `get_usdc_balance`
- **News**: `search_news`, `get_sentiment_analysis`
- **GitHub** (for CI/CD agents): `create_issue`, `list_pull_requests`, `analyze_repo`

**Why So Many Tools?**: LangChain agents need granular, single-purpose tools. Each tool does ONE thing well.

---

### `langchain/domain_tools.py`
**Purpose**: **Auto-generates LangChain tools from domain registry**  
**What It Does**: For each registered domain, creates tools dynamically

**Generated Tools**:
```python
# For "crypto" domain:
scan_crypto_markets = wrap_tool(...)       # Calls crypto scanner
analyze_crypto_market = wrap_tool(...)     # Calls crypto agent

# For "nba" domain:
scan_nba_markets = wrap_tool(...)
analyze_nba_game = wrap_tool(...)
```

**How It Works**:
```python
def get_domain_tools(domain_name: str) -> list[StructuredTool]:
    config = get_domain(domain_name)
    scanner = config.create_scanner(get_context())
    
    # Auto-wrap scanner methods
    tools = [
        wrap_tool(scanner.scan, name=f"scan_{domain_name}_markets"),
        wrap_tool(scanner.filter_tradeable, name=f"filter_{domain_name}"),
    ]
    return tools
```

**Benefit**: Add a new domain → get LangChain tools for free!

---

### `langchain/agent.py`
**Purpose**: **Pre-built LangChain agents** (convenience functions)  
**What It Does**: Factory functions for common agent patterns

**Available Agents**:
1. `create_crypto_agent()` - For crypto markets
2. `create_sports_agent()` - For NBA/sports markets
3. `create_probability_extraction_agent()` - Extract probabilities from text
4. `compare_ml_vs_market_forecast()` - Benchmark ML model vs market prices
5. `analyze_business_risks()` - For small business forecasting

**Example**:
```python
from polymarket_agents.langchain.agent import create_crypto_agent

agent = create_crypto_agent()
result = agent.invoke("Find BTC markets with >5% edge")
```

---

### `langchain/clob_tools.py`
**Purpose**: **Trading tools** for CLOB (order book)  
**What It Does**: Wraps Polymarket trading functions as LangChain tools

**Available Tools**:
- `get_orderbook` - Fetch bid/ask prices
- `place_order` - Submit limit order
- `get_usdc_balance` - Check wallet balance
- `get_position` - Current holdings in a market

---

## 5. 🕸️ **Graph** (LangGraph Agents)

### `graph/state.py`
**Purpose**: **Shared state for LangGraph agents**  
**What It Does**: Defines state schema for multi-step agent workflows

**Key Concept**: LangGraph agents pass state between nodes. This defines that state structure.

```python
@dataclass
class AgentState:
    messages: list[BaseMessage]      # Conversation history
    market_id: Optional[str]          # Current market being analyzed
    research_data: Optional[dict]     # Fetched data
    recommendation: Optional[str]     # Final output
```

---

### `graph/memory_agent.py`
**Purpose**: **LangGraph agent with memory persistence**  
**What It Does**: Multi-turn conversation agent that remembers context

**Architecture**:
```
┌──────────┐
│  START   │
└────┬─────┘
     │
     v
┌─────────────┐      ┌──────────────┐
│  Research   │─────▶│  Analysis    │
│    Node     │      │    Node      │
└─────────────┘      └──────┬───────┘
                            │
                            v
                      ┌─────────────┐
                      │ Recommend   │
                      │    Node     │
                      └──────┬──────┘
                             │
                             v
                        ┌────────┐
                        │  END   │
                        └────────┘
```

**Nodes**:
1. **Research**: Fetch market data, news, price history
2. **Analysis**: Run ML models, calculate edge
3. **Recommend**: Format output, save to database

**Use Case**: Multi-step market analysis with intermediate checkpoints.

---

### `graph/planning_agent.py`
**Purpose**: **Planning agent with reflection**  
**What It Does**: Creates execution plan, runs steps, reflects on results

**Pattern**: ReAct (Reasoning + Acting)
```
1. Plan: Break task into steps
2. Act: Execute first step
3. Observe: Check results
4. Reflect: Adjust plan if needed
5. Repeat until done
```

---

## 6. 🧠 **ML Foundations** (Neural Networks from Scratch)

### `ml_foundations/nn.py`
**Purpose**: **Pure NumPy neural network** (no TensorFlow/PyTorch!)  
**What It Does**: Implements feedforward neural network for probability estimation

**Why Build from Scratch?**: Educational + full control over architecture.

**Key Class: `NeuralNetwork`**
```python
nn = NeuralNetwork(layer_sizes=[10, 20, 20, 1])  # Input, hidden, hidden, output
nn.train(X_train, y_train, epochs=1000, learning_rate=0.01)
predictions = nn.predict(X_test)
```

**Implemented Features**:
- Forward propagation
- Backpropagation (gradient descent)
- Activation functions (sigmoid, ReLU, tanh)
- He/Xavier weight initialization
- Mini-batch training

**Use Case**: Train probability models on market features (volume, spread, momentum).

---

### `ml_foundations/utils.py`
**Purpose**: **Math utilities** for ML (statistics, metrics)  
**What It Does**: Helper functions for probability calculations

**Functions**:
- `calculate_sharpe_ratio()` - Risk-adjusted returns
- `calculate_kelly_fraction()` - Position sizing
- `log_loss()` - Probability calibration metric
- `brier_score()` - Forecast accuracy

---

## 7. 🤖 **ML Strategies** (Prediction Models)

### `ml_strategies/base_strategy.py`
**Purpose**: **Base class for all ML strategies**  
**What It Does**: Defines common interface for prediction models

```python
class BaseStrategy(ABC):
    @abstractmethod
    def fit(self, X, y):
        """Train on historical data."""
        
    @abstractmethod
    def predict_proba(self, X) -> float:
        """Return probability (0.0 to 1.0)."""
```

---

### `ml_strategies/market_prediction.py`
**Purpose**: **Main predictor class** (ensemble of models)  
**What It Does**: Combines multiple models for robust predictions

**Models Used**:
- XGBoost (gradient boosting)
- LSTM (time series)
- KNN (nearest neighbors)
- Simple momentum (baseline)

**Usage**:
```python
predictor = MarketPredictor()
predictor.fit(historical_data)
prob = predictor.predict_market(market_id)
```

---

### `ml_strategies/xgboost_strategy.py`
**Purpose**: **XGBoost model wrapper**  
**What It Does**: Gradient boosting for tabular market features

**Features Used**:
- Volume (log-scaled)
- Spread (bid-ask)
- Price momentum (7d, 30d)
- Volatility
- Time to expiration

---

### `ml_strategies/lstm_probability.py`
**Purpose**: **LSTM for time series forecasting**  
**What It Does**: Predicts probability sequences using price history

**Architecture**:
```
Input: [price_t-30, price_t-29, ..., price_t]  (30 timesteps)
       ↓
  LSTM Layer (64 units)
       ↓
  Dropout (0.2)
       ↓
  LSTM Layer (32 units)
       ↓
  Dense (1 unit, sigmoid)
       ↓
Output: probability
```

---

### `ml_strategies/knn_strategy.py`
**Purpose**: **K-Nearest Neighbors baseline**  
**What It Does**: Finds similar historical markets, averages their outcomes

**Why KNN?**: Simple, interpretable, good for sparse data.

---

### `ml_strategies/simple_momentum.py`
**Purpose**: **Momentum-based strategies** (technical analysis)  
**What It Does**: Calculates trend-following signals

**Strategies**:
1. **Price Momentum**: Buy if price up >X% in last N days
2. **Volume Spike**: Buy if volume >X stddevs above average
3. **Mean Reversion**: Buy if price <X stddevs below mean

**Use Case**: Baseline strategies to beat.

---

### `ml_strategies/edge_detection.py`
**Purpose**: **Finds markets with statistical edge**  
**What It Does**: Compares ML predictions vs market prices, flags mispriced markets

```python
detector = EdgeDetector()
edges = detector.find_edges(markets, min_edge=0.05)

for edge in edges:
    print(f"{edge.market.question}: {edge.edge:.1%} edge")
```

---

### `ml_strategies/evaluation.py`
**Purpose**: **Model performance tracking**  
**What It Does**: Calculates accuracy, calibration, ROI metrics

**Metrics**:
- **Accuracy**: % correct predictions
- **Brier Score**: Probability calibration (0 = perfect)
- **Log Loss**: Penalizes confident wrong predictions
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Worst losing streak

---

### `ml_strategies/registry.py`
**Purpose**: **Strategy registry** (like domain registry)  
**What It Does**: Pluggable ML model system

```python
register_strategy("xgboost", XGBoostStrategy)
register_strategy("lstm", LSTMStrategy)

strategy = get_strategy("xgboost")
```

---

## 8. 🔬 **AutoML** (Automated ML Pipeline)

### `automl/auto_ml_pipeline.py`
**Purpose**: **End-to-end AutoML orchestrator**  
**What It Does**: Automates data ingestion → training → evaluation → deployment

**Pipeline Steps**:
1. **Data Ingestion**: Fetch historical markets from database
2. **Data Quality**: Validate, clean, handle missing values
3. **Feature Engineering**: Create predictive features
4. **Model Training**: Train multiple models in parallel
5. **Hyperparameter Tuning**: Grid search for best params
6. **Evaluation**: Test on holdout set
7. **Deployment**: Save best model, log to MLflow

**Usage**:
```python
pipeline = AutoMLPipeline(experiment_name="btc_forecasts")
pipeline.run(
    models_to_train=["xgboost", "lstm", "knn"],
    days_back=365,
    min_volume=1000
)
```

---

### `automl/data_ingestion.py`
**Purpose**: **Fetch and prepare training data**  
**What It Does**: Queries database, builds feature matrix

**Output Format**:
```python
{
    "X": pd.DataFrame([...]),  # Features
    "y": pd.Series([...]),      # Labels (0 or 1)
    "metadata": {...}           # Market IDs, dates
}
```

---

### `automl/data_quality.py`
**Purpose**: **Data validation and cleaning**  
**What It Does**:
- Detects missing values
- Identifies outliers (Z-score, IQR)
- Checks for data drift (distribution shifts)
- Validates feature distributions

**Why Critical?**: Garbage in = garbage out. Bad data breaks models.

---

### `automl/ml_database.py`
**Purpose**: **ML-specific database operations**  
**What It Does**: Stores model artifacts, experiments, predictions

**Tables**:
- `experiments`: Track hyperparameters, metrics
- `predictions`: Store all predictions for backtesting
- `models`: Serialized model blobs

---

### `automl/ml_tools.py` ⚠️ **COMPLEX FILE (765 lines)**
**Purpose**: **LangChain tools for AutoML** (agents can train models!)  
**What It Does**: Wraps AutoML pipeline as LangChain tools

**Available Tools**:
- `DataIngestionTool` - Fetch training data
- `ModelTrainingTool` - Train a specific model
- `PredictionTool` - Generate predictions
- `EvaluationTool` - Calculate metrics
- `AutoMLPipelineTool` - Run full pipeline

**Use Case**: LLM agent that trains ML models on demand.

```python
agent = create_agent_with_tools([AutoMLPipelineTool()])
result = agent.invoke("Train XGBoost on BTC markets from last year")
```

---

### `automl/ml_agent.py`
**Purpose**: **ML research agent** (autonomous model development)  
**What It Does**: Agent that experiments with features, models, hyperparameters

**Capabilities**:
- Suggests new features to try
- Compares model architectures
- Tunes hyperparameters
- Analyzes feature importance
- Writes experiment reports

**Pattern**: Agent-driven AutoML (like AutoGluon/H2O, but with LLM reasoning).

---

## 9. 💾 **Memory** (Persistence Layer)

### `memory/manager.py`
**Purpose**: **Central database interface** (SQLite)  
**What It Does**: All database operations (CRUD for markets, bets, experiments)

**Key Class: `MemoryManager`**
```python
db = MemoryManager("data/markets.db")

# Market operations
db.add_market(market_data)
markets = db.get_top_markets(limit=10, category="crypto")
db.update_market_price(market_id, new_price)

# Bet tracking
bet_id = db.record_bet(market_id, side="YES", amount=100, price=0.55)
db.update_bet_result(bet_id, outcome="WIN", profit=82)

# Research logging
db.add_research(market_id, source="newsapi", data={"articles": [...]})

# Agent execution tracking
execution_id = db.start_agent_execution(agent_name="crypto_agent")
db.add_node_execution(execution_id, node="research", status="success")
db.end_agent_execution(execution_id, result="Found 3 edges")
```

**Database Schema** (12 tables):
- `markets`: All markets (live and historical)
- `price_snapshots`: Timestamped price history
- `bets`: Placed bets with outcomes
- `research_data`: News, analysis, model outputs
- `agent_executions`: LangGraph run logs
- `experiments`: AutoML experiments
- `predictions`: Model predictions for backtesting
- ... (see schema in file)

**Type Safety**: Uses `Optional[str]` return types, `assert` guards for non-null constraints.

---

## 10. 🛠️ **Services**

### `services/ingestion.py`
**Purpose**: **Background data ingestion service**  
**What It Does**: Scheduled jobs for data collection

**Jobs**:
1. **Market Refresh**: Fetch new markets every 15 minutes
2. **Price Updates**: Update crypto prices every 5 minutes
3. **NBA Stats**: Fetch game stats daily
4. **Resolution Check**: Check if markets resolved

**Deployment**: Run as cron job or systemd service.

---

## 11. 🤝 **Subagents** (Specialized Agents)

### `subagents/market_research.py`
**Purpose**: **Research-focused subagent**  
**What It Does**: Deep dive on a specific market (news, social, technical)

### `subagents/risk_analysis.py`
**Purpose**: **Risk assessment subagent**  
**What It Does**: Analyzes portfolio risk, position sizing, stop losses

### `subagents/performance_monitor.py`
**Purpose**: **Portfolio tracking subagent**  
**What It Does**: Tracks P&L, win rate, Sharpe ratio

### `subagents/strategy_dev.py`
**Purpose**: **Strategy development subagent**  
**What It Does**: Backtests new strategies, writes reports

### `subagents/github_agent.py`
**Purpose**: **GitHub integration agent**  
**What It Does**: Creates issues, analyzes repo, suggests improvements

**Use Case**: CI/CD pipelines that run prediction models and open GitHub issues if accuracy drops.

### `subagents/github_ml_agent.py`
**Purpose**: **ML-focused GitHub agent**  
**What It Does**: Analyzes ML experiment results, suggests model improvements

### `subagents/data_collection.py`
**Purpose**: **Data collection coordinator**  
**What It Does**: Orchestrates multiple data sources (prices, news, stats)

---

## 12. 🧰 **Tools** (Granular Functions)

### `tools/market_tools.py`
**Purpose**: **Market-specific helper functions**  
**What It Does**: Parse market questions, extract entities, validate data

### `tools/trade_tools.py`
**Purpose**: **Trading utilities**  
**What It Does**: Position sizing, order validation, slippage calculation

### `tools/research_tools.py`
**Purpose**: **Research utilities**  
**What It Does**: Fetch documentation, summarize articles, sentiment analysis

### `tools/github_tools.py`
**Purpose**: **GitHub API wrappers**  
**What It Does**: Create issues, list PRs, fetch commit history

### `tools/gamma_markets.py`
**Purpose**: **Gamma API-specific tools**  
**What It Does**: Advanced queries (tag filtering, event grouping)

---

## 13. 🧩 **Utils** (Helpers)

### `utils/database.py`
**Purpose**: **Database utilities** (migrations, backups)

### `utils/analytics.py`
**Purpose**: **Analytics functions** (metrics, visualizations)

### `utils/text.py`
**Purpose**: **Text processing** (cleaning, tokenization)

### `utils/structures.py`
**Purpose**: **Data structures** (custom collections, trees)

### `utils/objects.py`
**Purpose**: **Pydantic models** (Market, Event, Article, Tag)

**Key Models**:
```python
class Market(BaseModel):
    id: str
    question: str
    outcomePrices: str  # JSON string
    volume: float
    
class PolymarketEvent(BaseModel):
    id: str
    title: str
    markets: list[Market]
    tags: list[Tag]
```

### `utils/context.py`
**Purpose**: **Context utilities** (similar to main `context.py` but domain-specific)

### `utils/utils.py`
**Purpose**: **Miscellaneous helpers** (date parsing, formatting)

---

## 14. 📦 **Other Files**

### `py.typed`
**Purpose**: **PEP 561 marker** (enables mypy type checking)  
**Content**: Single comment: `# Marker file for PEP 561`

**Why Important**: Tells mypy this package has inline type annotations.

---

## 🔗 How It All Connects

### **Typical Workflow: Crypto Market Analysis**

```
1. User: "Find profitable BTC markets"
   ↓
2. LangChain agent receives query
   ↓
3. Agent calls: scan_crypto_markets tool
   ↓
4. Tool → domains.registry.get_domain("crypto")
   ↓
5. CryptoScanner.scan() → Gamma API
   ↓
6. Returns 50 BTC markets
   ↓
7. CryptoScanner.enrich() → ccxt price API
   ↓
8. Adds current prices, volatility
   ↓
9. CryptoScanner.filter_tradeable()
   ↓
10. Filters to 5 markets (volume >$5k, edge >5%)
    ↓
11. ML model (XGBoost) predicts probabilities
    ↓
12. EdgeDetector compares ML prob vs market prob
    ↓
13. Kelly criterion calculates position sizes
    ↓
14. CryptoAgent builds recommendations
    ↓
15. MemoryManager saves to database
    ↓
16. Agent formats response to user
```

---

## 🎯 Key Architectural Patterns

### 1. **Dependency Injection**
- **Problem**: Hardcoded paths break tests
- **Solution**: `context.py` with injectable dependencies

### 2. **Plugin Architecture**
- **Problem**: Adding new market types requires touching many files
- **Solution**: Domain registry with factory pattern

### 3. **Protocol-Based Design**
- **Problem**: Concrete classes create tight coupling
- **Solution**: `EventScanner`, `PriceDataSource` protocols

### 4. **Tool Wrapping**
- **Problem**: LangChain dependency in every file
- **Solution**: `wrap_tool()` abstraction with fallback

### 5. **Separation of Concerns**
- **Connectors**: API clients (no business logic)
- **Domains**: Business logic (no API details)
- **Tools**: LangChain adapters (no ML code)
- **ML Strategies**: Prediction models (no API calls)

---

## 📊 File Complexity Matrix

| Module | Files | Lines | Complexity | Key File |
|--------|-------|-------|------------|----------|
| `langchain/` | 5 | ~2500 | HIGH | `tools.py` (1492 lines) |
| `automl/` | 7 | ~2000 | HIGH | `ml_tools.py` (765 lines) |
| `domains/` | 15 | ~1800 | MEDIUM | `crypto/scanner.py` |
| `memory/` | 2 | ~800 | MEDIUM | `manager.py` |
| `connectors/` | 6 | ~1200 | LOW | `polymarket.py` |
| `ml_strategies/` | 12 | ~2500 | MEDIUM | `market_prediction.py` |
| `graph/` | 3 | ~600 | MEDIUM | `memory_agent.py` |
| `utils/` | 8 | ~1000 | LOW | Various helpers |

---

## 🚀 Getting Started as a Developer

### **If You Want to...**

#### **Add a New Market Type (e.g., NFL)**
1. Create `domains/nfl/` directory
2. Define `NFLMarket(Market)` in `models.py`
3. Implement `NFLScanner(EventScanner)` in `scanner.py`
4. Write `NFLAgent` in `agent.py`
5. Register in `domains/registry.py`

#### **Add a New ML Model**
1. Create `ml_strategies/my_model.py`
2. Inherit from `BaseStrategy`
3. Implement `fit()` and `predict_proba()`
4. Register in `ml_strategies/registry.py`

#### **Add a New LangChain Tool**
1. Write implementation function in `langchain/tools.py`
2. Define Pydantic schema
3. Wrap with `wrap_tool()`
4. Test with `tool.invoke({"param": value})`

#### **Add a New Data Source**
1. Create `connectors/my_api.py`
2. Implement client class
3. Add to `context.py` as injectable dependency
4. Use via `get_context().my_api`

---

## 🎓 Learning Paths

### **Beginner**: Understanding the System
1. Start with `config.py` (environment setup)
2. Read `context.py` (dependency injection)
3. Explore `connectors/gamma.py` (API client)
4. Study `domains/base.py` (core abstractions)
5. Run `examples/` scripts

### **Intermediate**: Building Features
1. Study `domains/crypto/agent.py` (full pipeline)
2. Read `langchain/tools.py` (tool patterns)
3. Explore `memory/manager.py` (database operations)
4. Implement a new tool
5. Add a new domain scanner

### **Advanced**: ML & Architecture
1. Deep dive into `automl/auto_ml_pipeline.py`
2. Study `ml_foundations/nn.py` (neural network from scratch)
3. Read `ml_strategies/edge_detection.py` (edge calculation)
4. Implement custom ML strategy
5. Build new AutoML feature

---

## 🔍 Quick Reference

### **Import Patterns**
```python
# Configuration
from polymarket_agents.config import OPENAI_API_KEY, DEFAULT_MODEL

# Context (DI)
from polymarket_agents.context import get_context, set_context

# Connectors
from polymarket_agents.connectors.gamma import GammaMarketClient
from polymarket_agents.connectors.polymarket import Polymarket

# Domains
from polymarket_agents.domains.registry import get_domain, list_domains
from polymarket_agents.domains.crypto import CryptoAgent

# LangChain
from polymarket_agents.langchain.tools import get_top_volume_markets
from polymarket_agents.langchain.agent import create_crypto_agent

# Memory
from polymarket_agents.memory.manager import MemoryManager

# ML
from polymarket_agents.ml_strategies.market_prediction import MarketPredictor
from polymarket_agents.automl.auto_ml_pipeline import AutoMLPipeline
```

---

## 📝 TODOs & Future Work

### **Missing Features** (inferred from code structure)
- [ ] `api/` directory is empty (REST API for web UI?)
- [ ] `application/` has trading logic but no deployment scripts
- [ ] More domain types (politics, tech, climate)
- [ ] Real-time model serving (FastAPI endpoint?)
- [ ] Backtesting framework with paper trading
- [ ] Dashboard for monitoring agents
- [ ] Alerting system (Telegram/Discord bot)

### **Code Quality Improvements**
- [ ] Add docstrings to all public functions
- [ ] Write integration tests for full pipelines
- [ ] Set up CI/CD (GitHub Actions already started)
- [ ] Profile slow functions (LSTM training, database queries)
- [ ] Add logging throughout (structured JSON logs)
- [ ] Document deployment (Docker, systemd)

---

## 🎉 Summary

This is a **production-grade ML forecasting system** with:
- ✅ Clean architecture (DI, protocols, registries)
- ✅ Full type safety (mypy, pandas-stubs)
- ✅ Extensible design (plugin domains, ML strategies)
- ✅ Agent orchestration (LangChain, LangGraph)
- ✅ Real trading capabilities (Polymarket CLOB)
- ✅ Comprehensive data pipeline (ingestion → training → prediction)

**Total Lines of Code**: ~15,000 (excluding tests)  
**Complexity**: Production-ready with room for polish  
**Best For**: Small business ML forecasting, algorithmic prediction markets

---

**Questions? Dive into specific files or ask about patterns!**


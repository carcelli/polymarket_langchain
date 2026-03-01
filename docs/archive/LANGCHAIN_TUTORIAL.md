# 🎓 LangChain + ML Trading Agent Tutorial

> **Learn by exploring your own production codebase!**
> 
> You've already built a sophisticated LangChain + ML system for Polymarket.
> This tutorial shows you what you have and how to extend it.

---

## 🎉 **What You've Already Built**

### ✅ **8 Pre-Built LangChain Agents**

Your `langchain/agent.py` contains production-ready agents:

```python
1. create_polymarket_agent()           # General-purpose market analyst
2. create_structured_probability_agent()  # Returns typed forecasts
3. create_probability_extraction_agent()  # Extracts probabilities from text
4. create_research_agent()             # Deep research with RAG
5. create_ml_forecast_comparison_agent()  # ML vs market benchmarking
6. create_crypto_agent()               # Crypto-specific analysis
7. create_sports_agent()               # NBA markets
8. create_business_domain_agent()      # Small business forecasting
```

### ✅ **50+ LangChain Tools**

Your `langchain/tools.py` has wrapped every database/API function:

**Market Tools** (12 tools):
- `get_top_volume_markets` - Find high-volume markets
- `search_markets_db` - Full-text search
- `get_markets_by_category` - Filter by sports/politics/crypto
- `get_market_details` - Deep dive on specific market
- `get_price_history` - Time series data
- ... and 7 more

**Trading Tools** (8 tools):
- `get_orderbook` - Bid/ask prices
- `calculate_position_size` - Kelly criterion
- `get_usdc_balance` - Wallet balance
- `simulate_trade` - Paper trading
- ... and 4 more

**ML Tools** (10 tools):
- `DataIngestionTool` - Fetch training data
- `ModelTrainingTool` - Train XGBoost/LSTM
- `PredictionTool` - Generate forecasts
- `EvaluationTool` - Calculate metrics
- `AutoMLPipelineTool` - Full pipeline
- ... and 5 more

**Research Tools** (8 tools):
- `search_news` - NewsAPI integration
- `search_web` - Tavily web search
- `get_sentiment_analysis` - Text sentiment
- `fetch_documentation` - Research papers
- ... and 4 more

---

## 📚 **LangChain Patterns You're Already Using**

### **Pattern 1: Tool Wrapping** (from `tooling.py`)

You've abstracted LangChain dependency with graceful fallback:

```python
# Your code (tooling.py):
def wrap_tool(func, *, name=None, description=None, args_schema=None):
    """Wrap function as LangChain tool or minimal fallback."""
    if LANGCHAIN_AVAILABLE:
        return StructuredTool.from_function(func, ...)
    return ToolWrapper(func, ...)  # Minimal fallback
```

**Why This Works**: You can use tools with or without LangChain installed.

---

### **Pattern 2: Pydantic Schemas for Validation**

Every tool has type-safe inputs:

```python
# Your code (langchain/tools.py):
class TopVolumeMarketsInput(BaseModel):
    """Schema for getting top volume markets."""
    limit: int = Field(default=10, description="Maximum number of markets")
    category: Optional[str] = Field(None, description="Filter by category")

get_top_volume_markets = wrap_tool(
    _get_top_volume_markets_impl,
    name="get_top_volume_markets",
    args_schema=TopVolumeMarketsInput  # ← Type validation!
)
```

**Why This Works**: LangChain validates inputs before calling your function.

---

### **Pattern 3: ReAct Agent (Reason + Act)**

Your `create_polymarket_agent()` uses the ReAct pattern:

```python
# Your code (langchain/agent.py):
system_message = """You are an expert Polymarket trader.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat N times)
Thought: I now know the final answer
Final Answer: the final answer
"""

agent = create_react_agent(llm, tools, prompt)
```

**Why This Works**: The LLM iteratively reasons and acts until it solves the problem.

---

### **Pattern 4: Structured Output**

Your `create_structured_probability_agent()` returns typed objects:

```python
# Your code (langchain/agent.py):
class MarketForecast(BaseModel):
    market_id: str
    probability: float  # 0.0 to 1.0
    confidence: str     # "low", "medium", "high"
    reasoning: str
    data_sources: list[str]

structured_llm = llm.with_structured_output(MarketForecast, method="json_mode")
```

**Why This Works**: Get Python objects, not prose. Perfect for dashboards and automated trading.

---

### **Pattern 5: ML as Tools**

Your `automl/ml_tools.py` wraps ML models as LangChain tools:

```python
# Your code (automl/ml_tools.py):
class PredictionTool(BaseTool):
    """Generate probability predictions for a market using trained ML models."""
    
    name: str = "predict_market_probability"
    description: str = """Predict probability (0-1) for a market outcome.
    Input: market_id (str)
    Returns: JSON with probability, confidence, features used"""
    
    def _run(self, market_id: str, ...) -> str:
        # Load model
        model = self._load_model(model_id)
        
        # Fetch features
        features = self._prepare_features(market_id)
        
        # Predict
        probability = model.predict_proba(features)
        
        return json.dumps({
            "probability": float(probability),
            "confidence": self._calculate_confidence(features),
            "model": model_type
        })
```

**Why This Works**: LLM agents can invoke ML models dynamically based on reasoning.

---

## 🚀 **Hands-On: Your First LangChain Agent**

Let's create a simple agent using your existing infrastructure:

### **Step 1: Simple Market Research Agent** (5 minutes)

Create `examples/my_first_agent.py`:

```python
"""My first LangChain agent using existing infrastructure."""

from polymarket_agents.langchain.agent import create_simple_analyst

# Create agent (uses your 50+ tools automatically)
agent = create_simple_analyst(model="gpt-4o-mini")

# Ask a question
result = agent.invoke({
    "input": "Find the top 3 crypto markets by volume and summarize each"
})

print(result["output"])
```

**Run it:**
```bash
export OPENAI_API_KEY="your-key"
python examples/my_first_agent.py
```

**What happens:**
1. Agent calls `get_top_volume_markets(category="crypto", limit=3)`
2. For each market, calls `get_market_details(market_id)`
3. Synthesizes summary using LLM
4. Returns formatted answer

---

### **Step 2: Add Custom Tool** (10 minutes)

Let's add a sentiment analysis tool:

Create `my_tools/sentiment_tool.py`:

```python
"""Custom sentiment analysis tool."""

from polymarket_agents.tooling import wrap_tool
from pydantic import BaseModel, Field
from typing import Optional

def _analyze_market_sentiment_impl(question: str) -> str:
    """Analyze sentiment of market question using keyword matching.
    
    In production, replace with HuggingFace BERT or OpenAI embeddings.
    """
    positive_words = ["win", "succeed", "beat", "above", "increase", "pass"]
    negative_words = ["lose", "fail", "below", "decrease", "reject"]
    
    question_lower = question.lower()
    
    pos_count = sum(1 for word in positive_words if word in question_lower)
    neg_count = sum(1 for word in negative_words if word in question_lower)
    
    if pos_count > neg_count:
        sentiment = "bullish"
        score = 0.6 + (pos_count - neg_count) * 0.1
    elif neg_count > pos_count:
        sentiment = "bearish"
        score = 0.4 - (neg_count - pos_count) * 0.1
    else:
        sentiment = "neutral"
        score = 0.5
    
    return f'{{"sentiment": "{sentiment}", "score": {score:.2f}}}'


class SentimentInput(BaseModel):
    """Input schema for sentiment analysis."""
    question: str = Field(description="Market question to analyze")


# Wrap as LangChain tool
analyze_market_sentiment = wrap_tool(
    _analyze_market_sentiment_impl,
    name="analyze_market_sentiment",
    description="Analyze sentiment of a market question. Returns sentiment (bullish/bearish/neutral) and score (0-1).",
    args_schema=SentimentInput
)
```

**Use it in an agent:**

```python
from polymarket_agents.langchain.agent import create_polymarket_agent
from my_tools.sentiment_tool import analyze_market_sentiment

# Add your custom tool
custom_tools = [analyze_market_sentiment]

agent = create_polymarket_agent(
    model="gpt-4o-mini",
    tools=custom_tools,  # Only use your tool
    max_iterations=5
)

result = agent.invoke({
    "input": "What's the sentiment of 'Will BTC hit $100k in 2026?'"
})

print(result["output"])
```

---

### **Step 3: ML Integration** (20 minutes)

Use your existing ML tools in an agent:

```python
"""Agent that uses ML models for predictions."""

from polymarket_agents.langchain.agent import create_polymarket_agent
from polymarket_agents.automl.ml_tools import (
    DataIngestionTool,
    ModelTrainingTool,
    PredictionTool,
    EvaluationTool
)

# Create agent with ML tools
ml_tools = [
    DataIngestionTool(),
    PredictionTool(),
    EvaluationTool()
]

agent = create_polymarket_agent(
    model="gpt-4o",
    tools=ml_tools,
    max_iterations=10,
    verbose=True  # See reasoning steps
)

# Ask ML questions
query = """
1. Fetch training data for BTC markets from the last 6 months
2. Use the XGBoost model to predict probability for market ID 12345
3. Compare the prediction to the current market price
4. Calculate expected value if we bet YES
"""

result = agent.invoke({"input": query})
print(result["output"])
```

**What happens:**
1. Agent calls `DataIngestionTool` → fetches historical BTC markets
2. Calls `PredictionTool(model_id="xgboost")` → gets ML probability
3. Uses built-in math to calculate expected value: `EV = (prob * payout) - (1-prob * loss)`
4. Recommends trade if EV > 0

---

### **Step 4: Domain-Specific Agent** (15 minutes)

Your codebase has a **plugin architecture** for domains. Let's use it:

```python
"""Use domain-specific agents."""

from polymarket_agents.domains.registry import get_domain, list_domains
from polymarket_agents.context import get_context

# List available domains
print("Available domains:", list_domains())
# Output: ['crypto', 'nba']

# Get crypto domain
crypto_domain = get_domain("crypto")

# Create agent
crypto_agent = crypto_domain.create_agent(get_context())

# Run analysis
recommendations = crypto_agent.run()

for rec in recommendations:
    print(f"\n{rec.market.question}")
    print(f"  Our Prediction: {rec.edge.our_prob:.1%}")
    print(f"  Market Price:   {rec.edge.market_prob:.1%}")
    print(f"  Edge:           {rec.edge.edge:+.1%}")
    print(f"  Action:         {rec.action}")
    print(f"  Size:           {rec.size_fraction:.1%} of bankroll")
    print(f"  Reasoning:      {rec.reasoning}")
```

**How it works:**
1. `CryptoAgent.run()` → scans Polymarket for BTC/ETH markets
2. Enriches with live prices from `ccxt` exchanges
3. Filters to high-volume, high-liquidity markets
4. Calculates edge (your prob vs market prob)
5. Uses Kelly criterion for position sizing
6. Returns structured recommendations

---

## 🧠 **LangChain + ML Patterns in Your Codebase**

### **Pattern 1: ML Model as Tool**

```python
# From automl/ml_tools.py
class PredictionTool(BaseTool):
    """ML model wrapped as LangChain tool."""
    
    def _run(self, market_id: str, model_type: str = "xgboost") -> str:
        # 1. Load trained model
        model = load_model(model_type)
        
        # 2. Prepare features
        features = self._get_market_features(market_id)
        
        # 3. Predict
        probability = model.predict_proba(features)
        
        # 4. Return as JSON string (LangChain requirement)
        return json.dumps({
            "probability": float(probability),
            "model": model_type,
            "features_used": list(features.keys())
        })
```

**Key Insight**: ML models become **first-class tools** the LLM can reason about.

---

### **Pattern 2: Agent-Driven AutoML**

```python
# From automl/ml_agent.py (your codebase)
class MLResearchAgent:
    """Agent that experiments with ML models autonomously."""
    
    def __init__(self):
        self.agent = create_polymarket_agent(
            tools=[
                DataIngestionTool(),
                ModelTrainingTool(),
                HyperparameterTuningTool(),
                EvaluationTool(),
                FeatureImportanceTool()
            ]
        )
    
    def optimize_model(self, target_metric: str = "brier_score"):
        """Let agent optimize ML pipeline."""
        query = f"""
        Your goal: Minimize {target_metric} on BTC price prediction markets.
        
        Steps:
        1. Ingest historical data (last 12 months)
        2. Train baseline XGBoost model
        3. Evaluate on holdout set
        4. Try 3 hyperparameter combinations
        5. Report best model and metrics
        """
        
        return self.agent.invoke({"input": query})
```

**Key Insight**: The LLM decides **which experiments to run** based on intermediate results.

---

### **Pattern 3: Ensemble Reasoning**

```python
# From langchain/agent.py (your codebase)
def create_ml_forecast_comparison_agent():
    """Agent that compares ML predictions vs market prices."""
    
    tools = [
        get_market_price_tool,      # Current market probability
        predict_with_xgboost_tool,  # ML prediction 1
        predict_with_lstm_tool,     # ML prediction 2
        predict_with_knn_tool,      # ML prediction 3
        calculate_ensemble_tool,    # Weighted average
        calculate_edge_tool         # ML vs market
    ]
    
    system_prompt = """
    For each market:
    1. Fetch current market price (implied probability)
    2. Get predictions from XGBoost, LSTM, KNN
    3. Calculate ensemble (weighted average by historical accuracy)
    4. Compare ensemble to market price
    5. If |ensemble - market| > 5%, flag as mispriced
    """
    
    return create_polymarket_agent(tools=tools, prompt=system_prompt)
```

**Key Insight**: Agent orchestrates **multiple ML models** and synthesizes results.

---

### **Pattern 4: RAG for Market Research**

```python
# From langchain/agent.py (your codebase)
def create_research_agent():
    """Agent with retrieval-augmented generation for market analysis."""
    
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    # Vector DB with historical market data
    vectorstore = Chroma(
        persist_directory="data/chroma",
        embedding_function=OpenAIEmbeddings()
    )
    
    retrieval_tool = create_retriever_tool(
        vectorstore.as_retriever(),
        "search_historical_markets",
        "Search historical resolved markets for similar patterns"
    )
    
    tools = [
        retrieval_tool,              # Semantic search
        get_market_details,          # Current data
        search_news,                 # Recent articles
        predict_with_ml              # ML forecast
    ]
    
    return create_polymarket_agent(tools=tools)
```

**Key Insight**: Agent combines **semantic search** (past patterns) with **live data** and **ML**.

---

## 🎯 **Real-World Use Case: Autonomous Trading**

Your `application/trade.py` shows the full pipeline:

```python
# From application/trade.py (your actual code)
class Trader:
    def one_best_trade(self):
        """Find and execute best trade autonomously."""
        
        # 1. Fetch all tradeable events
        events = self.polymarket.get_all_tradeable_events()
        
        # 2. Filter with RAG (semantic search for relevant markets)
        filtered_events = self.agent.filter_events_with_rag(events)
        
        # 3. Get markets for each event
        markets = self.agent.map_filtered_events_to_markets(filtered_events)
        
        # 4. Filter by liquidity, volume
        tradeable_markets = self.agent.filter_markets(markets)
        
        # 5. Find best trade (ML + LLM reasoning)
        best_trade = self.agent.source_best_trade(tradeable_markets[0])
        
        # 6. Execute (commented for safety)
        # self.polymarket.execute_market_order(market, amount)
        
        return best_trade
```

**Flow:**
```
Events (API) 
  → RAG filter (semantic) 
  → Markets (data) 
  → Liquidity filter (rules) 
  → ML prediction (XGBoost/LSTM) 
  → LLM reasoning (edge calculation) 
  → Trade recommendation (structured output)
  → Execution (CLOB)
```

---

## 🧪 **Exercises for Learning**

### **Exercise 1: Debug Agent Reasoning** (15 min)

Run an agent with `verbose=True` to see the ReAct loop:

```python
from polymarket_agents.langchain.agent import create_polymarket_agent

agent = create_polymarket_agent(verbose=True, max_iterations=5)

result = agent.invoke({
    "input": "Find BTC markets with volume >$10k and calculate expected value"
})
```

**You'll see:**
```
> Entering new AgentExecutor chain...
Thought: I need to find Bitcoin markets first
Action: search_markets_db
Action Input: {"query": "BTC Bitcoin", "min_volume": 10000}
Observation: [{"id": "12345", "question": "Will BTC hit $100k?", ...}]

Thought: Now I need the current price
Action: get_market_details
Action Input: {"market_id": "12345"}
Observation: {"yes_price": 0.65, "no_price": 0.35, ...}

Thought: I should get an ML prediction
Action: predict_market_probability
Action Input: {"market_id": "12345", "model_type": "xgboost"}
Observation: {"probability": 0.72, "confidence": "high"}

Thought: I can now calculate expected value
Expected value = (0.72 * 0.35) - (0.28 * 0.65) = +0.07 (7% edge)
Final Answer: Market 12345 has +7% expected value. Recommend BUY YES.
```

---

### **Exercise 2: Build a Backtesting Agent** (30 min)

Create an agent that tests strategies on historical data:

```python
"""Backtesting agent using your existing tools."""

from polymarket_agents.langchain.agent import create_polymarket_agent
from polymarket_agents.memory.manager import MemoryManager

# Get resolved markets
db = MemoryManager("data/markets.db")
resolved_markets = db.get_resolved_markets(days_back=90)

# Create agent
backtest_agent = create_polymarket_agent(
    model="gpt-4o",
    tools=[...],  # Your ML tools
    max_iterations=20
)

# Run backtest
query = f"""
Backtest the XGBoost strategy on {len(resolved_markets)} resolved markets.

For each market:
1. Get the closing price (before resolution)
2. Get ML prediction from XGBoost
3. Calculate if we would have won
4. Track P&L

Report:
- Win rate
- Average edge captured
- Sharpe ratio
- Max drawdown
"""

result = backtest_agent.invoke({"input": query})
```

---

### **Exercise 3: Multi-Agent System** (45 min)

Coordinate multiple specialized agents:

```python
"""Multi-agent system for trading."""

from polymarket_agents.langchain.agent import (
    create_research_agent,
    create_ml_forecast_comparison_agent,
    create_polymarket_agent
)

class TradingSystem:
    def __init__(self):
        self.research_agent = create_research_agent()
        self.ml_agent = create_ml_forecast_comparison_agent()
        self.execution_agent = create_polymarket_agent()
    
    def analyze_market(self, market_id: str):
        """Three-stage analysis with specialized agents."""
        
        # Stage 1: Research
        research = self.research_agent.invoke({
            "input": f"Research market {market_id}: news, sentiment, historical patterns"
        })
        
        # Stage 2: ML Prediction
        ml_forecast = self.ml_agent.invoke({
            "input": f"Compare ML models for market {market_id}, return ensemble probability"
        })
        
        # Stage 3: Execution Decision
        decision = self.execution_agent.invoke({
            "input": f"""
            Research summary: {research['output']}
            ML forecast: {ml_forecast['output']}
            
            Decide: BUY YES, BUY NO, or PASS
            If buy, calculate position size (Kelly criterion)
            """
        })
        
        return decision
```

---

## 📊 **Advanced Patterns in Your Codebase**

### **1. Context Engineering** (`utils/context.py`)

You have a sophisticated context management system:

```python
# From utils/context.py (your code)
class ContextManager:
    """Manages contextual information for agent prompts."""
    
    def get_model_context(self, market_focus: Optional[str] = None) -> str:
        """Generate context block for LLM prompts."""
        context = []
        
        # Add market statistics
        stats = self._get_market_stats()
        context.append(f"Active markets: {stats['active_count']}")
        context.append(f"Total volume 24h: ${stats['volume_24h']:,.2f}")
        
        # Add domain-specific context
        if market_focus == "crypto":
            prices = self._get_crypto_prices()
            context.append(f"BTC: ${prices['BTC']:,.0f}")
            context.append(f"ETH: ${prices['ETH']:,.0f}")
        
        return "\n".join(context)
```

**Use it:**
```python
from polymarket_agents.utils.context import ContextManager

ctx_manager = ContextManager()
agent = create_polymarket_agent(context_manager=ctx_manager)

# Agent now has live market stats in system prompt!
```

---

### **2. Memory Persistence** (`graph/memory_agent.py`)

Your LangGraph agents persist conversation history:

```python
# From graph/memory_agent.py (your code)
from langgraph.checkpoint.sqlite import SqliteSaver

# Persistent checkpoints
memory = SqliteSaver.from_conn_string("data/agent_memory.db")

agent = create_polymarket_agent(
    checkpointer=memory,
    thread_id="trading_session_001"
)

# Conversation history persists across runs
agent.invoke({"input": "Find BTC markets"})
agent.invoke({"input": "Which one has the most volume?"})  # Remembers context!
```

---

### **3. Async Agents** (`core/async_utils.py`)

Your `TaskSupervisor` enables parallel agent execution:

```python
# From core/async_utils.py (your code)
import asyncio
from polymarket_agents.core.async_utils import TaskSupervisor

supervisor = TaskSupervisor()

# Run multiple agents in parallel
await supervisor.start_task(
    "crypto_scanner",
    crypto_agent.run_async(),
    restart_on_failure=True
)

await supervisor.start_task(
    "nba_scanner",
    nba_agent.run_async(),
    restart_on_failure=True
)

# Monitor health
metrics = supervisor.get_metrics()
print(f"Crypto agent restarts: {metrics['crypto_scanner'].restarts}")
```

---

## 🚀 **Next Steps**

### **Week 1: Master Existing Tools**
- [ ] Run all 8 pre-built agents
- [ ] Understand each tool category (market, trading, ML, research)
- [ ] Trace ReAct reasoning with `verbose=True`
- [ ] Read `langchain/agent.py` line-by-line

### **Week 2: Build Custom Tools**
- [ ] Add sentiment analysis tool (HuggingFace)
- [ ] Add social media scraper tool
- [ ] Add technical indicator tool (RSI, MACD)
- [ ] Test tools in isolation, then add to agent

### **Week 3: ML Integration**
- [ ] Train XGBoost on historical markets
- [ ] Wrap as LangChain tool
- [ ] Create ensemble of 3 models
- [ ] Build backtesting agent

### **Week 4: Production Deployment**
- [ ] Add monitoring (metrics, alerts)
- [ ] Implement paper trading mode
- [ ] Set up async task supervisor
- [ ] Deploy with risk controls

---

## 🔗 **Resources**

### **Your Codebase**
- `langchain/agent.py` - 8 pre-built agents
- `langchain/tools.py` - 50+ tools
- `automl/ml_tools.py` - ML as tools
- `application/trade.py` - End-to-end trading bot
- `examples/` - Runnable demos

### **External**
- LangChain docs: https://python.langchain.com/docs/tutorials/agents
- LangGraph: https://langchain-ai.github.io/langgraph/
- Polymarket API: https://docs.polymarket.com/
- ReAct paper: https://arxiv.org/abs/2210.03629

---

## ⚠️ **Safety Checklist**

Before deploying with real money:

- [ ] Backtest on 1000+ resolved markets
- [ ] Paper trade for 30 days
- [ ] Set max position size (e.g., 2% of bankroll)
- [ ] Implement stop losses
- [ ] Monitor Sharpe ratio daily
- [ ] Track max drawdown
- [ ] Set up alerts for anomalies
- [ ] Use testnet first (Polygon Amoy)
- [ ] Review TOS: polymarket.com/tos
- [ ] Consult legal/financial advisor

---

## 🎓 **Bonus: LangChain Concepts Map**

```
LangChain Components (Ontology):

Objects:
- Prompts (templates)
- LLMs (language models)
- Tools (functions)
- Documents (text chunks)
- Retrievers (search)
- Memory (state)

Relations:
- Chains (sequential)
- Agents (dynamic routing)
- Graphs (LangGraph state machines)

Leverage Points:
- ReAct pattern (reasoning + acting)
- RAG (retrieval-augmented generation)
- Structured output (typed responses)
- Tool calling (function execution)
```

---

**You've already built a production system. Now master it!** 🚀


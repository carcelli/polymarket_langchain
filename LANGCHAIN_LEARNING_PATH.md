# 🎓 Your LangChain Learning Path

> **You've already built a production-grade LangChain + ML trading system.**
> This guide shows you how to master it.

---

## 🎯 **Your Goal**

Learn LangChain fundamentals, ML integration, and autonomous agent patterns by exploring your own sophisticated codebase.

---

## ✅ **What You Have (Summary)**

### **Production System Components**

```
┌─────────────────────────────────────────────────────┐
│  8 Pre-Built LangChain Agents (langchain/agent.py) │
├─────────────────────────────────────────────────────┤
│  • create_polymarket_agent()      ← General analyst │
│  • create_ml_forecast_comparison  ← ML benchmarking │
│  • create_crypto_agent()          ← BTC/ETH focus   │
│  • create_sports_agent()          ← NBA markets     │
│  • create_research_agent()        ← RAG-powered     │
│  • ... 3 more specialized agents                    │
└─────────────────────────────────────────────────────┘
          ↓ Uses
┌─────────────────────────────────────────────────────┐
│  50+ LangChain Tools (langchain/tools.py)           │
├─────────────────────────────────────────────────────┤
│  Market Tools:    search, filter, analyze (12)      │
│  Trading Tools:   orderbook, position size (8)      │
│  ML Tools:        train, predict, evaluate (10)     │
│  Research Tools:  news, web search, sentiment (8)   │
│  GitHub Tools:    issues, PRs, analysis (4)         │
│  ... and 8 more categories                          │
└─────────────────────────────────────────────────────┘
          ↓ Calls
┌─────────────────────────────────────────────────────┐
│  Data Layer (connectors/ + memory/)                 │
├─────────────────────────────────────────────────────┤
│  • Polymarket Gamma API (market data)               │
│  • CLOB API (trading execution)                     │
│  • NewsAPI (market research)                        │
│  • Tavily (web search)                              │
│  • SQLite Database (12 tables)                      │
│  • ccxt (crypto price feeds)                        │
└─────────────────────────────────────────────────────┘
          ↓ Integrates with
┌─────────────────────────────────────────────────────┐
│  ML Layer (automl/ + ml_strategies/)                │
├─────────────────────────────────────────────────────┤
│  • XGBoost (gradient boosting)                      │
│  • LSTM (time series)                               │
│  • KNN (nearest neighbors)                          │
│  • Custom Neural Network (NumPy)                    │
│  • AutoML Pipeline                                  │
└─────────────────────────────────────────────────────┘
```

---

## 📚 **Learning Path (4 Weeks)**

### **Week 1: LangChain Fundamentals** ⭐

**Goal**: Understand tools, agents, and ReAct pattern

#### Day 1-2: Run Existing Agents
```bash
cd /home/orson-dev/projects/polymarket_langchain

# Set up environment
export OPENAI_API_KEY="your-key-here"

# Run quickstart
python examples/langchain_quickstart.py

# Try each example (1-6)
```

**What to observe:**
- How tools are called (Action → Observation)
- How the agent decides which tool to use (Thought)
- How structured output works (typed Python objects)

#### Day 3-4: Read Core Files
- `langchain/agent.py` (lines 1-200) - Agent creation
- `langchain/tools.py` (lines 1-100) - Tool wrapping pattern
- `tooling.py` (all 72 lines) - Abstraction layer

**Key concepts to understand:**
1. **Tool wrapping**: `wrap_tool(func, name, description, args_schema)`
2. **ReAct pattern**: Thought → Action → Observation loop
3. **Pydantic schemas**: Type-safe tool inputs

#### Day 5-7: Trace Agent Execution
```python
from polymarket_agents.langchain.agent import create_polymarket_agent

agent = create_polymarket_agent(verbose=True, max_iterations=5)

# Watch the reasoning loop
result = agent.invoke({
    "input": "Find BTC markets and calculate expected value"
})
```

**Exercise**: Draw the execution flow:
```
Human Query
    ↓
LLM (Thought: "I need market data")
    ↓
Action: get_top_volume_markets(category="crypto")
    ↓
Observation: [{"id": "12345", "question": "...", ...}]
    ↓
LLM (Thought: "Now I need current price")
    ↓
Action: get_market_details(market_id="12345")
    ↓
... (continues until final answer)
```

---

### **Week 2: ML Integration** ⭐⭐

**Goal**: Understand how ML models become LangChain tools

#### Day 8-10: Study ML Tools
Read these files:
- `automl/ml_tools.py` (lines 1-200) - ML as tools
- `ml_strategies/market_prediction.py` - Predictor class
- `ml_strategies/xgboost_strategy.py` - XGBoost wrapper

**Key pattern**:
```python
class PredictionTool(BaseTool):
    name = "predict_market_probability"
    description = "Predict probability using ML model"
    
    def _run(self, market_id: str, model_type: str) -> str:
        # 1. Load model
        model = load_model(model_type)
        
        # 2. Prepare features
        features = prepare_features(market_id)
        
        # 3. Predict
        prob = model.predict_proba(features)
        
        # 4. Return JSON string (LangChain requirement)
        return json.dumps({"probability": float(prob)})
```

#### Day 11-12: Build Custom ML Tool
```python
# my_ml_tools.py
from polymarket_agents.tooling import wrap_tool
from polymarket_agents.ml_strategies.xgboost_strategy import XGBoostStrategy
from pydantic import BaseModel, Field

class XGBoostInput(BaseModel):
    market_id: str = Field(description="Market to predict")

def xgboost_predict(market_id: str) -> str:
    """Predict using XGBoost model."""
    model = XGBoostStrategy()
    # Load pre-trained weights
    model.load("models/xgboost_btc.pkl")
    
    # Get features
    from polymarket_agents.memory.manager import MemoryManager
    db = MemoryManager()
    market = db.get_market(market_id)
    features = extract_features(market)
    
    # Predict
    prob = model.predict_proba(features)
    
    return f'{{"probability": {prob:.3f}, "model": "xgboost"}}'

# Wrap as tool
xgboost_tool = wrap_tool(
    xgboost_predict,
    name="predict_with_xgboost",
    args_schema=XGBoostInput
)
```

#### Day 13-14: Create ML Agent
```python
from polymarket_agents.langchain.agent import create_polymarket_agent
from my_ml_tools import xgboost_tool

agent = create_polymarket_agent(
    tools=[xgboost_tool],
    model="gpt-4o",
    max_iterations=10
)

result = agent.invoke({
    "input": """
    For market ID 12345:
    1. Get ML prediction
    2. Get current market price
    3. Calculate edge
    4. Recommend trade if edge > 5%
    """
})
```

---

### **Week 3: Advanced Patterns** ⭐⭐⭐

**Goal**: Master structured output, RAG, multi-agent systems

#### Day 15-17: Structured Output
Study `langchain/agent.py` lines 166-241:

```python
# Your code already does this!
class MarketForecast(BaseModel):
    market_id: str
    probability: float
    confidence: str
    reasoning: str

structured_llm = llm.with_structured_output(MarketForecast)

# Now you get Python objects, not strings!
forecast = structured_llm.invoke("Analyze BTC market 12345")
print(f"Probability: {forecast.probability:.1%}")
```

**Exercise**: Create your own structured output schema:
```python
class TradingSignal(BaseModel):
    action: Literal["BUY_YES", "BUY_NO", "PASS"]
    confidence: float  # 0-1
    position_size: float  # % of bankroll
    reasoning: str
    expected_value: float

trading_llm = llm.with_structured_output(TradingSignal)
```

#### Day 18-19: RAG (Retrieval-Augmented Generation)
Study `langchain/agent.py` lines 242-396:

Your `create_research_agent()` uses vector search over historical markets:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Vector DB with past market outcomes
vectorstore = Chroma(
    persist_directory="data/chroma",
    embedding_function=OpenAIEmbeddings()
)

# Semantic search tool
retrieval_tool = create_retriever_tool(
    vectorstore.as_retriever(),
    "search_historical_markets",
    "Find similar resolved markets"
)
```

**Exercise**: Build your own RAG system:
1. Embed all resolved markets into Chroma
2. Create retrieval tool
3. Agent queries: "Find markets similar to 'Will BTC hit $100k?'"
4. Use results to inform current prediction

#### Day 20-21: Multi-Agent Systems
```python
# Coordinate specialized agents
class TradingPipeline:
    def __init__(self):
        self.researcher = create_research_agent()
        self.ml_agent = create_ml_forecast_comparison_agent()
        self.trader = create_polymarket_agent()
    
    def analyze(self, market_id):
        # Stage 1: Research
        context = self.researcher.invoke({
            "input": f"Research market {market_id}"
        })
        
        # Stage 2: ML Forecast
        forecast = self.ml_agent.invoke({
            "input": f"Predict for {market_id}"
        })
        
        # Stage 3: Decision
        decision = self.trader.invoke({
            "input": f"Context: {context}\nForecast: {forecast}\nDecide: trade or pass?"
        })
        
        return decision
```

---

### **Week 4: Production Deployment** ⭐⭐⭐⭐

**Goal**: Deploy safely with monitoring and risk controls

#### Day 22-24: Async Agents
Study `core/async_utils.py`:

```python
from polymarket_agents.core.async_utils import TaskSupervisor
import asyncio

supervisor = TaskSupervisor()

# Run multiple agents in parallel
async def run_agents():
    await supervisor.start_task(
        "crypto_monitor",
        crypto_agent.run_async(),
        restart_on_failure=True
    )
    
    await supervisor.start_task(
        "nba_monitor",
        nba_agent.run_async(),
        restart_on_failure=True
    )
    
    # Monitor health
    while True:
        metrics = supervisor.get_metrics()
        if metrics["crypto_monitor"].errors > 10:
            await supervisor.stop_task("crypto_monitor")
            # Alert ops team
        
        await asyncio.sleep(60)

asyncio.run(run_agents())
```

#### Day 25-26: Backtesting
```python
from polymarket_agents.memory.manager import MemoryManager

db = MemoryManager()
resolved = db.get_resolved_markets(days_back=90)

results = []
for market in resolved:
    # Get closing price (before resolution)
    price = market["close_price"]
    
    # Get ML prediction
    pred = model.predict(market["features"])
    
    # Would we have traded?
    if abs(pred - price) > 0.05:  # 5% edge threshold
        side = "YES" if pred > price else "NO"
        outcome = market["resolution"]
        won = (side == "YES" and outcome == "YES") or (side == "NO" and outcome == "NO")
        results.append({
            "market": market["id"],
            "won": won,
            "edge": pred - price
        })

# Calculate metrics
win_rate = sum(r["won"] for r in results) / len(results)
avg_edge = sum(r["edge"] for r in results) / len(results)
print(f"Win rate: {win_rate:.1%}")
print(f"Avg edge captured: {avg_edge:.1%}")
```

#### Day 27-28: Risk Controls
```python
class SafeTrader:
    """Trading bot with safety guardrails."""
    
    def __init__(self):
        self.agent = create_polymarket_agent()
        self.max_position_pct = 0.02  # Max 2% of bankroll
        self.max_daily_loss_pct = 0.10  # Stop if down 10% in a day
        self.min_liquidity = 5000  # Skip illiquid markets
        
        self.daily_pnl = 0
        self.positions = {}
    
    def should_trade(self, recommendation):
        """Safety checks before trading."""
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss_pct:
            return False, "Daily loss limit reached"
        
        # Check liquidity
        if recommendation.market.liquidity < self.min_liquidity:
            return False, "Insufficient liquidity"
        
        # Check position size
        if recommendation.size_fraction > self.max_position_pct:
            return False, "Position size too large"
        
        # Check edge threshold
        if abs(recommendation.edge.edge) < 0.05:
            return False, "Edge too small (<5%)"
        
        return True, "OK"
    
    def execute_safely(self):
        """Execute with safety checks."""
        recs = self.agent.run()
        
        for rec in recs:
            should_trade, reason = self.should_trade(rec)
            
            if should_trade:
                # Paper trade first!
                self.simulate_trade(rec)
                # self.execute_real_trade(rec)  # Uncomment after 30 days paper trading
            else:
                print(f"Skipping {rec.market.question}: {reason}")
```

---

## 🎯 **Milestones**

### **Milestone 1: Understanding** (Week 1)
- [ ] Run all 6 quickstart examples
- [ ] Trace a ReAct loop manually
- [ ] Understand tool wrapping pattern
- [ ] Read `langchain/agent.py` (first 200 lines)

### **Milestone 2: Building** (Week 2)
- [ ] Create custom tool with Pydantic schema
- [ ] Wrap ML model as tool
- [ ] Build agent with custom tools
- [ ] Train and deploy ML model

### **Milestone 3: Mastery** (Week 3)
- [ ] Implement structured output for trading signals
- [ ] Build RAG system with vector search
- [ ] Create multi-agent pipeline
- [ ] Backtest on 100+ resolved markets

### **Milestone 4: Production** (Week 4)
- [ ] Deploy with async task supervisor
- [ ] Add monitoring and alerts
- [ ] Implement risk controls
- [ ] Paper trade for 30 days
- [ ] Calculate real-world metrics (Sharpe, drawdown)

---

## 📊 **Success Metrics**

Track these to measure your learning:

### **Technical Mastery**
- [ ] Can explain ReAct pattern without notes
- [ ] Can wrap any Python function as LangChain tool
- [ ] Can create structured output schemas
- [ ] Can build RAG system from scratch
- [ ] Can deploy async multi-agent system

### **ML Integration**
- [ ] Can wrap scikit-learn model as tool
- [ ] Can create ensemble of 3+ models
- [ ] Can backtest strategy on historical data
- [ ] Can calculate Brier score, Sharpe ratio
- [ ] Can explain when to use each ML model

### **Production Readiness**
- [ ] Agent runs 24/7 without crashes
- [ ] Error rate <1%
- [ ] Restarts <5 per day
- [ ] Response time <30 seconds
- [ ] Backtested on 1000+ markets
- [ ] Paper traded 30+ days
- [ ] Sharpe ratio >1.5
- [ ] Max drawdown <20%

---

## 🔗 **Key Files to Master**

Priority order:

1. **`langchain/agent.py`** (1055 lines)
   - 8 agent factories
   - ReAct prompts
   - Structured output

2. **`langchain/tools.py`** (1492 lines)
   - 50+ tools
   - Tool wrapping pattern
   - Pydantic schemas

3. **`automl/ml_tools.py`** (765 lines)
   - ML as tools
   - Training/prediction pipeline
   - Evaluation metrics

4. **`domains/crypto/agent.py`** (~400 lines)
   - Full pipeline example
   - Domain-specific logic
   - Edge calculation

5. **`core/async_utils.py`** (467 lines)
   - Task supervision
   - Error handling
   - Metrics tracking

6. **`application/trade.py`** (77 lines)
   - End-to-end trading bot
   - Shows full integration

---

## 🚀 **Quick Start (Right Now)**

```bash
cd /home/orson-dev/projects/polymarket_langchain

# 1. Set API key
export OPENAI_API_KEY="your-key"

# 2. Run quickstart
python examples/langchain_quickstart.py

# Choose option 5 to see ReAct reasoning

# 3. Read the tutorial
cat LANGCHAIN_TUTORIAL.md

# 4. Study your first agent
less langchain/agent.py  # Read create_polymarket_agent()

# 5. Try modifying an agent
# Open langchain/agent.py and change max_iterations from 10 to 3
# See how it affects behavior
```

---

## 📚 **Resources**

### **Your Codebase** (Best Resource!)
- `LANGCHAIN_TUTORIAL.md` - This tutorial
- `CODEBASE_BREAKDOWN.md` - Every file explained
- `CODEBASE_ESSENTIALS.md` - Top 10 files
- `examples/langchain_quickstart.py` - Hands-on demos
- `examples/` - More runnable examples

### **External**
- [LangChain Docs](https://python.langchain.com/docs/tutorials/agents) - Official tutorials
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/) - State machine agents
- [ReAct Paper](https://arxiv.org/abs/2210.03629) - Original research
- [Polymarket Docs](https://docs.polymarket.com/) - API reference

---

## 💡 **Pro Tips**

1. **Always run with `verbose=True` first** - Understand what the agent is doing
2. **Start simple** - One tool, one agent, one query
3. **Read tool descriptions** - They're the agent's "API docs"
4. **Trace execution manually** - Draw the Thought → Action → Observation loop
5. **Use structured output** - Get Python objects, not strings
6. **Backtest everything** - Never deploy without testing on historical data
7. **Paper trade first** - Run for 30 days before real money
8. **Monitor metrics** - Track Sharpe, drawdown, win rate daily
9. **Read your own code** - You've built something sophisticated!
10. **Start with quickstart** - Run `examples/langchain_quickstart.py` right now

---

## 🎉 **You're Ready!**

You have:
- ✅ 8 production agents
- ✅ 50+ tools
- ✅ ML integration
- ✅ Full trading infrastructure
- ✅ This learning path

**Next step: Run the quickstart!**

```bash
python examples/langchain_quickstart.py
```

---

**Learn by doing. You've already built the system. Now master it!** 🚀


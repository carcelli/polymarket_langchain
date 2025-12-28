# LangChain Integration Reference for Polymarket Agents

This document provides a comprehensive reference for integrating Polymarket API functionality with LangChain.

> **📚 See also:** [POLYMARKET_API_REFERENCE.md](./POLYMARKET_API_REFERENCE.md) for complete API endpoint documentation, including REST endpoints, WebSocket channels, rate limits, and data structures.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding Tool Arguments](#understanding-tool-arguments)
3. [Available Tools Reference](#available-tools-reference)
4. [Creating Custom Tools](#creating-custom-tools)
5. [Agent Patterns](#agent-patterns)
6. [Pydantic Schemas](#pydantic-schemas)
7. [Best Practices](#best-practices)
8. [Polymarket API Overview](#polymarket-api-overview)

---

## Quick Start

```python
from agents.langchain import get_all_tools
from agents.langchain.agent import create_polymarket_agent, run_agent

# Create agent with all tools
agent = create_polymarket_agent(
    model="gpt-4o-mini",
    temperature=0.1,
    max_iterations=10
)

# Run a query
result = run_agent(agent, "Find the best political market to trade")
print(result)
```

---

## Understanding Tool Arguments

### How LangChain Infers Arguments

LangChain's `@tool` decorator automatically infers:

1. **Tool Name**: From function name
2. **Description**: From function docstring
3. **Arguments**: From type hints
4. **Defaults**: From parameter defaults

### Example Tool Anatomy

```python
@tool
def fetch_markets(
    limit: int = 20,              # Inferred: int, default=20
    active_only: bool = True      # Inferred: bool, default=True
) -> str:
    """Fetch Polymarket markets.     <-- Tool description
    
    Use this to see available markets for trading.
    
    Args:
        limit: Maximum markets to return     <-- Arg descriptions
        active_only: Only return active markets
    
    Returns:
        JSON string of markets
    """
    ...
```

### Supported Argument Types

| Type | Example | Notes |
|------|---------|-------|
| `str` | `query: str` | Basic string |
| `int` | `limit: int = 10` | Integer with default |
| `float` | `price: float` | Decimal number |
| `bool` | `active: bool = True` | True/False |
| `Optional[str]` | `date: Optional[str] = None` | Nullable string |
| `List[str]` | `keywords: List[str]` | String array |

---

## Available Tools Reference

### 1. Market Tools

#### `fetch_all_markets`
Fetch all available Polymarket prediction markets.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `limit` | `int` | `20` | Maximum markets to return |

**Returns**: JSON string with market data including:
- `id`: Market identifier
- `question`: The prediction question
- `outcomes`: Available outcomes (e.g., Yes/No)
- `outcome_prices`: Current prices
- `active`: Trading status
- `spread`: Bid-ask spread

**Example**:
```python
from agents.langchain.tools import fetch_all_markets
result = fetch_all_markets.invoke({"limit": 5})
```

---

#### `fetch_tradeable_markets`
Fetch only active, tradeable markets (filters out closed/archived).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `limit` | `int` | `20` | Maximum markets |

---

#### `get_market_by_token`
Get detailed info for a specific market by token ID.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `token_id` | `str` | Yes | CLOB token ID (long numeric string) |

**Example token_id**: `"101669189743438912873361127612589311253202068943959811456820079057046819967115"`

---

#### `get_market_by_id`
Get detailed info for a specific market by its Gamma market ID.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `market_id` | `int` | Yes | Gamma market ID (integer, not CLOB token) |

**Note**: Use this when you have a Gamma market ID (integer). Use `get_market_by_token` for CLOB token IDs (long strings).

---

#### `get_current_markets_gamma`
Fetch markets from Gamma API with extended metadata.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `limit` | `int` | `10` | Maximum markets |

---

#### `get_clob_tradable_markets`
Fetch markets with active CLOB (limit order book) trading.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `limit` | `int` | `10` | Maximum markets |

---

### 2. Event Tools

#### `fetch_all_events`
Fetch Polymarket events (collections of related markets).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `limit` | `int` | `20` | Maximum events |

---

#### `fetch_tradeable_events`
Fetch only tradeable events (active, not restricted).

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `limit` | `int` | `20` | Maximum events |

---

#### `get_event_by_id`
Get detailed info for a specific event by its Gamma event ID.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `event_id` | `int` | Yes | Gamma event ID (integer) |

**Example**:
```python
from agents.langchain.tools import get_event_by_id
result = get_event_by_id.invoke({"event_id": 12345})
```

---

#### `get_current_events_gamma`
Fetch events from Gamma API with extended metadata.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `limit` | `int` | `10` | Maximum events |

---

### 3. Order Book Tools

#### `get_orderbook`
Get current order book for a market token.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `token_id` | `str` | Yes | CLOB token ID |

**Returns**: Bids and asks with price/size pairs

---

#### `get_orderbook_price`
Get current mid-market price for a token.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `token_id` | `str` | Yes | CLOB token ID |

---

### 4. Account Tools

#### `get_usdc_balance`
Get wallet's USDC balance. **No arguments.**

---

#### `get_wallet_address`
Get wallet address from private key. **No arguments.**

---

### 5. Research Tools

#### `search_news`
Search for news articles on a topic.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `keywords` | `str` | Yes | Comma-separated keywords |

**Example**: `"Trump,Biden,election"`

---

### 6. Analysis Tools

#### `get_superforecast`
Get superforecaster-style probability estimate.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `event_title` | `str` | Yes | Event description |
| `market_question` | `str` | Yes | Prediction question |
| `outcome` | `str` | Yes | Outcome to analyze ("Yes"/"No") |

**Example**:
```python
get_superforecast.invoke({
    "event_title": "2024 Presidential Election",
    "market_question": "Will Biden win?",
    "outcome": "Yes"
})
```

---

#### `analyze_market_with_llm`
Ask LLM to analyze markets based on your query.

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `user_query` | `str` | Yes | Natural language query |

---

### 7. RAG Tools

#### `query_markets_rag`
Query local vector database of markets.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `query` | `str` | Required | Search query |
| `db_directory` | `str` | `"./local_db_markets"` | Vector DB path |

---

#### `create_markets_rag_database`
Create/update local vector database of markets.

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `directory` | `str` | `"./local_db_markets"` | Where to store DB |

---

### 8. Trading Tools

#### `preview_order`
Preview an order (does NOT execute).

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `token_id` | `str` | Yes | CLOB token ID |
| `price` | `float` | Yes | Price (0.01-0.99) |
| `size` | `float` | Yes | Number of shares |
| `side` | `str` | Yes | "BUY" or "SELL" |

---

## Creating Custom Tools

### Method 1: @tool Decorator (Simple)

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(market_id: int, analyze: bool = True) -> str:
    """Analyze a specific market.
    
    Args:
        market_id: The market ID to analyze
        analyze: Whether to run full analysis
    
    Returns:
        Analysis results as JSON
    """
    # Your implementation
    return json.dumps({"market_id": market_id, "analyzed": analyze})
```

### Method 2: StructuredTool with Pydantic (Complex)

```python
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class TradeInput(BaseModel):
    """Input schema for trade tool."""
    token_id: str = Field(description="Market token ID")
    price: float = Field(ge=0.01, le=0.99, description="Limit price")
    size: float = Field(gt=0, description="Number of shares")
    side: str = Field(pattern="^(BUY|SELL)$", description="Trade direction")

def execute_trade_impl(token_id: str, price: float, size: float, side: str) -> str:
    # Implementation
    return "Trade executed"

execute_trade = StructuredTool.from_function(
    func=execute_trade_impl,
    name="execute_trade",
    description="Execute a trade on Polymarket",
    args_schema=TradeInput
)
```

### Method 3: Wrapping Existing Methods

```python
from agents.polymarket.polymarket import Polymarket

@tool
def get_balance() -> str:
    """Get current USDC balance."""
    poly = Polymarket()
    return f"Balance: {poly.get_usdc_balance():.2f} USDC"
```

### Tool Wrappers for Other Runtimes

Agent tools are defined as plain functions and wrapped via `wrap_tool`, so they
can also be used outside LangChain. For raw callables:

```python
from agents.langchain.tools import get_tool_functions

tool_funcs = get_tool_functions()
tool_funcs["fetch_all_markets"](limit=5)
```

CLOB tools follow the same pattern:

```python
from agents.langchain.clob_tools import get_clob_tool_functions

clob_funcs = get_clob_tool_functions()
clob_funcs["clob_get_market"](condition_id="...")
```

---

## Agent Patterns

### ReAct Agent (Recommended)

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
tools = get_all_tools()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Polymarket analyst. Use tools to analyze markets."),
    ("human", "{input}\n\n{agent_scratchpad}")
])

agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10)

result = executor.invoke({"input": "What markets should I trade?"})
```

### Tool-Calling Agent (Simpler)

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

llm = ChatOpenAI(model="gpt-4o-mini")
tools = get_market_tools()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You analyze prediction markets."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
```

### LangGraph Multi-Step (Advanced)

```python
from agents.langchain.agent import create_langgraph_trader

graph = create_langgraph_trader()
result = graph.invoke({
    "messages": [HumanMessage(content="Find the best trade")]
})
```

---

## Pydantic Schemas

### Pre-defined Schemas

```python
from agents.langchain.tools import (
    MarketQueryInput,
    EventQueryInput,
    OrderInput,
    NewsQueryInput,
    ForecastInput,
)

# MarketQueryInput
class MarketQueryInput(BaseModel):
    limit: int = Field(default=10, description="Max markets")
    active_only: bool = Field(default=True, description="Active only")

# OrderInput
class OrderInput(BaseModel):
    token_id: str = Field(description="CLOB token ID")
    price: float = Field(description="Price 0.01-0.99")
    size: float = Field(description="Number of shares")
    side: str = Field(description="BUY or SELL")
```

---

## Best Practices

### 1. Tool Docstrings Are Critical
The LLM uses docstrings to understand when/how to use tools.

```python
@tool
def good_tool(query: str) -> str:
    """Search for markets matching a query.
    
    Use this when you need to find markets about a specific topic.
    Returns market IDs and current prices.
    
    Args:
        query: Topic to search for (e.g., "election", "crypto")
    """
```

### 2. Handle Errors Gracefully
```python
@tool
def safe_tool(market_id: str) -> str:
    """Fetch market data safely."""
    try:
        return json.dumps(get_market(market_id))
    except Exception as e:
        return f"Error: {str(e)}"
```

### 3. Return Structured Data
```python
@tool
def structured_result() -> str:
    """Returns structured JSON that LLM can parse."""
    return json.dumps({
        "markets": [...],
        "total": 10,
        "status": "success"
    }, indent=2)
```

### 4. Limit Data Volume
```python
@tool
def limited_fetch(limit: int = 10) -> str:
    """Fetch markets with sensible limits."""
    # Don't return 1000s of markets
    markets = fetch_all()[:min(limit, 50)]
    return json.dumps(markets)
```

### 5. Use Lazy Initialization
```python
_client = None

def get_client():
    global _client
    if _client is None:
        _client = ExpensiveClient()
    return _client

@tool
def my_tool() -> str:
    client = get_client()  # Only initialized once
    return client.fetch()
```

---

## Environment Variables

Required in `.env`:

```bash
OPENAI_API_KEY="sk-..."           # Required for LLM
POLYGON_WALLET_PRIVATE_KEY="..."  # Required for trading
NEWSAPI_API_KEY="..."             # Optional for news
TAVILY_API_KEY="..."              # Optional for web search

# CLOB Client (py_clob_client) - Optional, derive from private key
CLOB_API_URL="https://clob.polymarket.com"  # CLOB API endpoint
CLOB_API_KEY="..."                # Optional - derived if not set
CLOB_SECRET="..."                 # Optional - derived if not set
CLOB_PASS_PHRASE="..."            # Optional - derived if not set
CHAIN_ID="137"                    # Polygon (137) or Amoy (80002)
```

---

## CLOB Tools Reference (py_clob_client)

The CLOB tools wrap the official `py_clob_client` library for direct CLOB API access.

### Quick Start with CLOB Tools

```python
from agents.langchain.clob_tools import (
    get_clob_market_tools,    # Read-only market data
    get_clob_trading_tools,   # Order execution
    get_clob_rfq_tools,       # RFQ (Request for Quote)
    print_clob_argument_reference,
)

# Read-only agent (no auth required)
tools = get_clob_market_tools()

# Full trading agent (requires credentials)
from agents.langchain.clob_tools import get_all_clob_tools
tools = get_all_clob_tools()

# Print argument reference
print_clob_argument_reference()
```

### CLOB Market Data Tools

| Tool | Args | Description |
|------|------|-------------|
| `clob_health_check` | None | Check API connectivity |
| `clob_get_server_time` | None | Get server timestamp |
| `clob_get_midpoint` | `token_id` | Mid-market price |
| `clob_get_price` | `token_id`, `side` | BUY/SELL price |
| `clob_get_orderbook` | `token_id` | Full order book |
| `clob_get_spread` | `token_id` | Bid-ask spread |
| `clob_get_last_trade_price` | `token_id` | Last executed trade |
| `clob_get_markets` | `next_cursor` | All markets (paginated) |
| `clob_get_simplified_markets` | `next_cursor` | Simplified market list |
| `clob_get_market` | `condition_id` | Single market details |

### CLOB Trading Tools

| Tool | Args | Description |
|------|------|-------------|
| `clob_create_limit_order` | `token_id`, `price`, `size`, `side` | Submit limit order |
| `clob_create_market_order` | `token_id`, `amount`, `side` | Submit market order |
| `clob_cancel_order` | `order_id` | Cancel single order |
| `clob_cancel_all_orders` | None | Cancel all orders |
| `clob_get_open_orders` | `market`, `asset_id` | List open orders |
| `clob_get_order` | `order_id` | Get order details |
| `clob_get_trades` | `market`, `maker_address` | Trade history |

### CLOB RFQ Tools

| Tool | Args | Description |
|------|------|-------------|
| `clob_create_rfq_request` | `token_id`, `price`, `size`, `side` | Create RFQ request |
| `clob_get_rfq_requests` | `state`, `limit` | List RFQ requests |
| `clob_create_rfq_quote` | `request_id`, `token_id`, `price`, `size`, `side` | Quote an RFQ |
| `clob_get_rfq_quotes` | `request_id`, `state`, `limit` | List quotes |
| `clob_accept_rfq_quote` | `request_id`, `quote_id`, `expiration_seconds` | Accept quote |
| `clob_cancel_rfq_request` | `request_id` | Cancel RFQ request |

### Example: CLOB Agent for Market Making

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

from agents.langchain.clob_tools import (
    clob_get_midpoint,
    clob_get_orderbook,
    clob_get_spread,
    clob_create_limit_order,
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
tools = [clob_get_midpoint, clob_get_orderbook, clob_get_spread, clob_create_limit_order]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a market maker. Analyze spreads and place orders."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)

result = executor.invoke({
    "input": "Check the spread for token 12345... and suggest a market-making strategy"
})
```

---

## Quick Reference Card

```python
# Import tools
from agents.langchain.tools import (
    get_all_tools,           # All 21 tools
    get_market_tools,        # Market fetching (6 tools)
    get_event_tools,         # Event fetching (4 tools)
    get_trading_tools,       # Trading/orderbook (5 tools)
    get_analysis_tools,      # Analysis/RAG (6 tools)
    get_read_only_tools,     # Market + Event tools (10 tools, no trading)
)

# Import agents
from agents.langchain.agent import (
    create_polymarket_agent,   # Full-featured agent
    create_simple_analyst,     # Analysis only
    create_research_agent,     # Research focused
    run_agent,                 # Execute query
    find_best_trade,           # Automated workflow
)

# Print argument reference
from agents.langchain.tools import print_argument_reference
print_argument_reference()
```

---

## Tool Collection Summary

### Agent Tools (polymarket-agents wrapper)

| Collection | Tools | Description |
|------------|-------|-------------|
| `get_all_tools()` | 21 | Complete toolset for full agent |
| `get_market_tools()` | 6 | Market fetching and lookup |
| `get_event_tools()` | 4 | Event fetching and lookup |
| `get_read_only_tools()` | 10 | Markets + Events (no trading) |
| `get_trading_tools()` | 5 | Orderbook, wallet, preview |
| `get_analysis_tools()` | 6 | Forecasting, RAG, news |

### CLOB Tools (py_clob_client wrapper)

| Collection | Tools | Description |
|------------|-------|-------------|
| `get_all_clob_tools()` | 25 | All CLOB tools |
| `get_clob_market_tools()` | 10 | Prices, orderbooks, spreads |
| `get_clob_trading_tools()` | 7 | Orders, cancellations |
| `get_clob_account_tools()` | 2 | Balances, API keys |
| `get_clob_rfq_tools()` | 6 | RFQ requests/quotes |
| `get_clob_readonly_tools()` | 10 | Market data only |

### Combined Tools

| Collection | Tools | Description |
|------------|-------|-------------|
| `get_combined_tools()` | 46 | All tools from both sources |
| `get_combined_readonly_tools()` | 20 | Read-only from both sources |

---

## Polymarket API Overview

This section provides a quick reference to the underlying Polymarket APIs. For complete documentation, see [POLYMARKET_API_REFERENCE.md](./POLYMARKET_API_REFERENCE.md).

### API Endpoints

| API | Base URL | Purpose |
|-----|----------|---------|
| **Gamma API** | `https://gamma-api.polymarket.com` | Market discovery, events, metadata |
| **CLOB API** | `https://clob.polymarket.com` | Trading, order management, order books |
| **Data API** | `https://data-api.polymarket.com` | User holdings, on-chain activity |
| **WebSocket** | `wss://ws-subscriptions-clob.polymarket.com/ws/` | Real-time order book, trades |

### Common Query Patterns

#### Fetch Active Markets
```bash
curl "https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=20"
```

#### Fetch Market by Slug (from URL)
```bash
curl "https://gamma-api.polymarket.com/events/slug/fed-decision-in-october"
```

#### Fetch by Tag/Category
```bash
curl "https://gamma-api.polymarket.com/events?tag_id=100381&closed=false"
```

#### Get Order Book
```bash
curl "https://clob.polymarket.com/book?token_id=101669..."
```

### Key IDs to Understand

| ID Type | Example | Where to Find |
|---------|---------|---------------|
| **Market ID** | `253123` | Gamma API response `id` field |
| **Event ID** | `12345` | Gamma API events response |
| **CLOB Token ID** | `101669189743438912873361127612589311253202068943959811456820079057046819967115` | Market `clobTokenIds` field |
| **Condition ID** | `0x26ee82bee...` | CLOB market identifier |
| **Slug** | `fed-decision-in-october` | URL path after `/event/` or `/market/` |

### Rate Limits

| Endpoint Type | Rate Limit |
|---------------|------------|
| Public REST | 10 requests/second |
| Authenticated REST | 100 requests/second |
| WebSocket | 100 messages/second |

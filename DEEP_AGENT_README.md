# Deep Research Agent Customization Guide

This document explains how to customize and use the deepagents framework integration in your Polymarket trading system for enhanced research capabilities.

## 🎯 What is DeepAgents?

DeepAgents is a framework that provides:
- **Automated Planning**: Agents break down complex tasks into manageable steps
- **File System Tools**: Persistent context management across conversations
- **Subagent Capabilities**: Delegate specialized tasks to focused subagents
- **Enhanced Tool Integration**: Seamless combination of multiple tool types

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  Deep Research   │───▶│  Trading Agent  │
│                 │    │     Agent        │    │   (Existing)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │   Research       │
                       │   Tools          │
                       │ • Web Search     │
                       │ • Market Data    │
                       │ • News Analysis  │
                       └──────────────────┘
```

## 🔧 Integration Points

### 1. Enhanced Research Tools (`agents/tools/research_tools.py`)

Added web search capabilities to complement your existing market database:

```python
from agents.tools.research_tools import web_search, comprehensive_research

# Web search for market context
results = web_search("Trump 2028 election polls", max_results=5)

# Comprehensive research package
research = comprehensive_research(market_id)
```

### 2. Deep Research Agent (`agents/deep_research_agent.py`)

New agent that combines all capabilities:

```python
from agents.deep_research_agent import analyze_market_with_deep_research

# Comprehensive analysis with web research
result = analyze_market_with_deep_research("Will Bitcoin reach $200k by 2025?")
print(result["analysis"])
```

### 3. Backward Compatibility

Your existing agents continue to work unchanged:

```python
# Existing planning agent still works
from agents.graph.planning_agent import analyze_bet
result = analyze_bet("Will Trump win 2028?")

# New deep agent for enhanced analysis
from agents.deep_research_agent import analyze_market_with_deep_research
deep_result = analyze_market_with_deep_research("Will Trump win 2028?")
```

## 🚀 Key Improvements

### Research Capabilities

| Feature | Traditional Agent | Deep Agent |
|---------|------------------|------------|
| Data Sources | Local market DB only | Web + markets + news |
| Research Depth | Structured pipeline | LLM-driven exploration |
| Context Management | In-memory state | File system persistence |
| Analysis Scope | Single market focus | Multi-market correlation |

### Planning & Execution

- **Automated Planning**: Deep agents automatically break down complex research tasks
- **Context Persistence**: Save research findings to files for future reference
- **Subagent Delegation**: Spawn specialized agents for different analysis aspects
- **Iterative Refinement**: Agents can revisit and improve their analysis

## 📋 Usage Examples

### Basic Market Analysis

```python
# Enhanced analysis with web research
from agents.deep_research_agent import analyze_market_with_deep_research

result = analyze_market_with_deep_research(
    "Will the Federal Reserve cut rates in 2025?"
)
print(result["analysis"])  # Comprehensive report with web insights
```

### Opportunity Scanning

```python
# Scan markets with deep research
from agents.deep_research_agent import scan_opportunities_with_deep_research

opportunities = scan_opportunities_with_deep_research(
    category="politics",
    min_volume=50000
)
print(opportunities["scan_results"])  # Ranked opportunities with research
```

### File-Based Context Management

The deep agent automatically uses file system tools to:
- Save research findings for future reference
- Manage large amounts of context data
- Build upon previous analysis work
- Share research across agent instances

## 🔧 Harness Capabilities

Following the deepagents harness documentation, we've implemented advanced agent capabilities:

### Storage Backends

**Filesystem Backend** (Default):
- Real filesystem access sandboxed to `./agent_workspace`
- Virtual mode prevents access outside the root directory
- Integrates with system tools (ripgrep for search)

**Composite Backend** (Recommended for research):
- Hybrid storage: `/` → FilesystemBackend, `/persistent/` → StoreBackend
- Temporary files in memory, persistent research in durable storage
- Longest-prefix routing for different storage strategies

**Store Backend** (Cross-conversation persistence):
- Uses LangGraph's BaseStore for durability
- Namespaced per assistant_id
- Files persist across conversations and sessions

### Subagent Delegation

**Specialized Subagents:**
- **market_researcher**: Handles web search, data gathering, and market analysis
- **risk_analyzer**: Focuses on risk assessment and Kelly criterion calculations
- **trade_executor**: Manages trade execution with verification (when trading enabled)

**Benefits:**
- Context isolation prevents main agent context bloat
- Parallel execution capabilities
- Specialized tools and prompts for different tasks
- Token efficiency through result compression

### Human-in-the-Loop

**Trading Approval System:**
- Interrupts execution before trade orders
- Shows trade details for human review
- Allows modification or cancellation of trades
- Safety gates for high-risk operations

**Configuration:**
```python
agent = create_polymarket_research_agent(
    enable_trading=True,
    enable_human_loop=True
)
```

### Large Result Eviction

**Automatic File System Offloading:**
- Monitors tool results for size (>20k tokens triggers eviction)
- Saves large results to files automatically
- Replaces with concise file references
- Prevents context window saturation

### Conversation History Summarization

**Automatic Context Management:**
- Triggers at 170,000 tokens
- Preserves recent 6 messages intact
- Older messages summarized by the model
- Enables very long conversations

### To-Do List Tracking

**Structured Task Management:**
- Built-in `write_todos` tool for task organization
- Tracks multiple tasks with status (pending, in_progress, completed)
- Persisted in agent state
- Helps organize complex multi-step research

### Prompt Caching (Anthropic)

**Performance Optimization:**
- Caches repeated prompt portions across turns
- ~10x speedup and cost reduction for long system prompts
- Automatically enabled for Claude models
- Transparent to agent operation

## 🧠 Long-Term Memory Capabilities

Following the deepagents long-term memory documentation, we've implemented comprehensive persistent memory across conversations:

### Memory Structure

**Hybrid Storage Architecture:**
- **Ephemeral Filesystem**: `/workspace/`, `/temp/`, `/cache/` (StateBackend)
- **Persistent Memory**: `/user/`, `/memories/`, `/knowledge/`, `/research/` (StoreBackend)
- **CompositeBackend Routing**: Automatic path-based storage selection

### Memory Organization

**User Data (`/user/` - StoreBackend):**
- `/user/preferences.txt` - Trading preferences and risk tolerance
- `/user/portfolio.txt` - Portfolio composition and constraints
- `/user/history.txt` - Past decisions and performance tracking

**Agent Memories (`/memories/` - StoreBackend):**
- `/memories/learnings.txt` - Accumulated insights and lessons
- `/memories/strategies.txt` - Successful trading strategies
- `/memories/context.txt` - Long-term conversation continuity
- `/memories/instructions.txt` - Self-improvement instructions

**Market Knowledge (`/knowledge/` - StoreBackend):**
- `/knowledge/markets/` - Market-specific understanding
- `/knowledge/patterns.txt` - Recognized market patterns
- `/knowledge/research/` - Research methodologies and frameworks

**Research Projects (`/research/` - StoreBackend):**
- `/research/active/` - Ongoing multi-session projects
- `/research/archive/` - Completed research with final reports

### Specialized Memory Agents

**Self-Improving Agent:**
```python
agent, store = create_self_improving_agent()
# Learns from user feedback, updates instructions over time
# Accumulates preferences in /user/preferences.txt
# Records successful strategies in /memories/strategies.txt
```

**Knowledge-Building Agent:**
```python
agent, store = create_knowledge_building_agent()
# Accumulates market understanding across conversations
# Builds pattern recognition in /knowledge/patterns.txt
# Develops domain expertise progressively
```

**Research Continuity Agent:**
```python
agent, store = create_research_continuity_agent()
# Maintains multi-session research projects
# Tracks progress in /research/active/[project_id]/
# Archives completed work in /research/archive/
```

### Cross-Thread Persistence

**Memory Access Across Conversations:**
```python
# Thread 1: Store preferences
agent.invoke({"messages": [{"role": "user", "content": "Save my conservative trading preferences"}]}, config1)

# Thread 2: Access same preferences
agent.invoke({"messages": [{"role": "user", "content": "What are my trading preferences?"}], config2)
# Agent reads from persistent /user/preferences.txt
```

### Memory Management Functions

**Initialization:**
```python
initialize_memory_structure(agent, store, thread_id)
# Creates complete directory structure and initial files
```

**Demonstration Functions:**
```python
demonstrate_cross_thread_memory()  # Shows persistence across threads
demonstrate_memory_accumulation()  # Shows progressive knowledge building
```

## 🔧 Advanced Backend Configurations

Following the deepagents backends documentation, we've implemented sophisticated storage strategies:

### Backend Strategies

**Filesystem Backend (Default)**:
- Sandboxed access to `./agent_workspace` with virtual mode
- Security policies blocking writes to `/secrets/`, `/system/`, `/config/`
- Path validation and symlink protection

**Composite Backend (Recommended)**:
```python
CompositeBackend(
    default=StateBackend(rt),  # Ephemeral by default
    routes={
        "/workspace/": FilesystemBackend(root="./agent_workspace", virtual_mode=True),
        "/research/": StoreBackend(rt, namespace="polymarket_research"),
        "/memories/": StoreBackend(rt, namespace="agent_memories"),
        "/temp/": StateBackend(rt),
        "/cache/": FilesystemBackend(root="./agent_cache", virtual_mode=True),
        "/market_data/": StoreBackend(rt, namespace="market_data"),
    }
)
```

**Store Backend (Persistent)**:
- Cross-conversation storage via LangGraph BaseStore
- Namespace isolation with security policies
- Access controls preventing writes to `/admin/`, `/system/`

**Virtual Backend (Demo/Testing)**:
- In-memory filesystem with pre-built research templates
- Market data structures and analysis frameworks
- No external dependencies, perfect for demos

### Security Policies

**Policy Enforcement**:
- Path-based access controls at the backend level
- Denied prefix restrictions (`/secrets/`, `/system/`, `/admin/`)
- Enterprise-grade security for sensitive operations

**Guarded Backends**:
```python
# Filesystem with policies
GuardedFilesystemBackend(
    root_dir="./secure_workspace",
    virtual_mode=True,
    deny_prefixes=["/secrets/", "/system/", "/config/"]
)

# Store with access controls
GuardedStoreBackend(
    runtime,
    namespace="secure_research",
    deny_prefixes=["/admin/", "/system/"]
)
```

### Advanced Routing Patterns

**Research Workflow Routing**:
- `/workspace/` → Active project files (Filesystem)
- `/research/` → Completed analyses (Store)
- `/memories/` → Agent learning (Store)
- `/temp/` → Intermediate results (State)

**Enterprise Setup**:
- `/user/` → User-specific data (Filesystem)
- `/shared/` → Team resources (Store)
- `/archive/` → Historical data (Store)
- `/cache/` → Performance optimization (Filesystem)

**Specialized Agents**:
```python
# Virtual filesystem for demos
virtual_demo_agent()

# Enterprise security setup
enterprise_secure_agent()

# Full routing capabilities
research_architect_agent()
```

## 🎭 Advanced Subagent Patterns

Following the deepagents subagents documentation, we've implemented sophisticated subagent coordination:

### Subagent Architecture

**Specialized Subagents:**
- **market_researcher**: Comprehensive market research with web search and data analysis
- **quick_researcher**: Fast answers for simple questions (1-2 searches)
- **risk_analyzer**: Quantitative risk assessment and Kelly criterion calculations
- **data_synthesizer**: Integration of multiple data sources into coherent analysis
- **trade_executor**: Precise trade execution with verification (when trading enabled)

### Context Isolation Patterns

**General-Purpose Subagent:**
```python
# Maximum context isolation for complex work
result = analyze_with_subagents(market_question, use_general_purpose=True)
# Agent delegates entire complex workflow to general-purpose subagent
```

**Direct Coordination:**
```python
# Main agent coordinates specialized subagents
result = analyze_with_subagents(market_question, use_general_purpose=False)
# market_researcher → risk_analyzer → data_synthesizer
```

**Research Team Workflow:**
```python
# Full team coordination pattern
result = research_team_analysis(market_question)
# Data Collector → Quantitative Analyzer → Synthesis Specialist
```

### Best Practices Implementation

**Clear Descriptions:**
```python
{
    "name": "market_researcher",
    "description": "Conducts comprehensive market research using web search and data analysis. Use for gathering information about market conditions, news, and expert opinions. Returns synthesized findings with confidence scores.",
    # ...
}
```

**Concise Result Formats:**
```python
"system_prompt": """...OUTPUT FORMAT (keep under 500 words):
## Research Summary (2-3 paragraphs)
[Synthesize key findings]

## Key Factors
• [Factor]: [Impact assessment]

## Confidence Assessment
• Data Quality: [High/Medium/Low]"""
```

**Appropriate Tool Sets:**
- `market_researcher`: web_search, market_news_search, comprehensive_research
- `risk_analyzer`: Built-in tools only (calculations focus)
- `trade_executor`: execute_market_order, execute_limit_order

### Context Bloat Prevention

**File-Based Storage:**
- Large research results saved to `/research/findings.md`
- Subagents return summaries, not raw data
- Main agent context stays focused on coordination

**Result Compression:**
- Subagent responses limited to 500 words
- Essential insights only, no intermediate data
- Clean handoffs between subagents

## 👥 Advanced Human-in-the-Loop

Following the deepagents human-in-the-loop documentation, we've implemented sophisticated approval workflows:

### Interrupt Configuration

**Risk-Based Decision Controls:**
```python
# Conservative: Maximum safety, no editing allowed
interrupts = {
    "execute_market_order": {"allowed_decisions": ["approve", "reject"]},
    "execute_limit_order": {"allowed_decisions": ["approve", "reject"]},
    "web_search": {"allowed_decisions": ["approve", "reject"]},  # API calls
}

# Moderate: Full control with editing
interrupts = {
    "execute_market_order": {"allowed_decisions": ["approve", "edit", "reject"]},
    "execute_limit_order": {"allowed_decisions": ["approve", "edit", "reject"]},
}

# Aggressive: Minimal interrupts
interrupts = {
    "read_file": False,  # No interrupts for safe operations
}
```

### Interrupt Handling Functions

**Complete Workflow Management:**
```python
from agents.deep_research_agent import (
    handle_agent_interrupt,
    create_human_decisions,
    resume_agent_with_decisions
)

# Check for interrupts
needs_approval, interrupt_info = handle_agent_interrupt(result, config)

if needs_approval:
    action_requests = interrupt_info["action_requests"]
    review_configs = interrupt_info["review_configs"]

    # Get user decisions
    user_decisions = get_user_approvals(action_requests, review_configs)

    # Create formatted decisions
    decisions = create_human_decisions(action_requests, review_configs, user_decisions)

    # Resume execution
    final_result = resume_agent_with_decisions(agent, decisions, config)
```

### Specialized Agent Functions

**Pre-Configured HITL Agents:**
```python
# Maximum safety with comprehensive approvals
trading_agent = create_trading_agent_with_approval()

# Conservative trading with restricted editing
conservative_agent = create_conservative_trading_agent()

# Full automation for research tasks
autonomous_agent = create_autonomous_research_agent()
```

### Interactive Sessions

**Complete Trading Workflows:**
```python
# Run interactive trading session with human oversight
interactive_trading_session()

# Features:
# 1. Agent analyzes market and proposes trades
# 2. Human reviews pending actions
# 3. Approve, edit, or reject decisions
# 4. Agent executes approved trades
```

### Multiple Tool Call Handling

**Batch Approval Processing:**
- All pending tool calls are batched in a single interrupt
- Decisions must match the order of `action_requests`
- Supports mixed approve/edit/reject decisions

### Argument Editing Capabilities

**Modify Before Execution:**
```python
# Edit trade parameters
edited_decision = {
    "type": "edit",
    "edited_action": {
        "name": "execute_market_order",
        "args": {
            "token_id": "modified_token",
            "amount": 750.0,  # Modified amount
            "side": "BUY"
        }
    }
}
```

### Checkpointer Integration

**State Persistence Requirements:**
- Human-in-the-loop requires `checkpointer` for state persistence
- Same `thread_id` must be used across interrupt/resume cycles
- Automatic state management for complex approval workflows

### Subagent Interrupt Policies

**Granular Control:**
- Main agent can have global interrupt policies
- Subagents can override with specialized policies
- `trade_executor` subagent requires approval for all trades
- Independent safety controls for different agent components

## 🎛️ Customization Options

Following the deepagents customization guide, we've implemented comprehensive configuration options:

### Model Configuration

```python
from langchain.chat_models import init_chat_model
from agents.deep_research_agent import create_polymarket_research_agent

# Using different models
claude_agent = create_polymarket_research_agent(
    model_name="claude-3-5-sonnet-20241022"
)

gpt_agent = create_polymarket_research_agent(
    model_name="gpt-4o"
)

# Or use LangChain model objects directly
from langchain_ollama import ChatOllama
custom_model = init_chat_model(ChatOllama(model="llama3.1", temperature=0.2))
agent = create_deep_agent(model=custom_model, tools=tools, system_prompt=prompt)
```

### Risk Tolerance Profiles

```python
# Conservative: High conviction required (5%+ edge, 10% max Kelly)
conservative_agent = create_polymarket_research_agent(risk_tolerance="conservative")

# Moderate: Balanced approach (3%+ edge, 15% max Kelly) - Default
moderate_agent = create_polymarket_research_agent(risk_tolerance="moderate")

# Aggressive: Lower thresholds (2%+ edge, 25% max Kelly)
aggressive_agent = create_polymarket_research_agent(risk_tolerance="aggressive")
```

### Trading Capabilities

```python
# Research-only agent (safe for analysis)
research_agent = create_polymarket_research_agent(enable_trading=False)

# Trading-enabled agent (can execute orders)
trading_agent = create_polymarket_research_agent(enable_trading=True)
```

### Convenience Functions

```python
from agents.deep_research_agent import (
    conservative_market_analysis,
    conservative_opportunity_scan,
    moderate_research_agent,
    trading_agent
)

# Quick conservative analysis
result = conservative_market_analysis("Will Bitcoin hit $200k?")

# Conservative opportunity scanning
opportunities = conservative_opportunity_scan("politics", limit=3)

# Get pre-configured agents
agent = moderate_research_agent()
result = agent.invoke({"messages": [{"role": "user", "content": "Analyze market..."}]})

# Harness-capable specialized agents
persistent_agent = persistent_research_agent()  # Cross-session memory
trading_agent = trading_agent_with_approval()   # Human approval required
high_perf_agent = high_performance_agent()      # All optimizations enabled

# Subagent coordination
result = analyze_with_subagents("Will Bitcoin hit $200k?")
```

## 🛠️ Setup Requirements

1. **Install Dependencies** (already done):
   ```bash
   pip install deepagents tavily-python
   ```

2. **Set API Keys**:
   ```bash
   export ANTHROPIC_API_KEY="your-anthropic-key"
   export TAVILY_API_KEY="your-tavily-key"
   ```

3. **Run Demo**:
   ```bash
   python demo_deep_agent.py
   ```

## 🎛️ Configuration Options

### Model Selection

Deep agents support multiple LLM providers:

```python
agent = create_deep_agent(
    tools=tools,
    system_prompt=custom_prompt,
    model="claude-3-5-sonnet-20241022"  # or "gpt-4o", "gemini-pro"
)
```

### Custom System Prompts

Tailor the agent for your specific trading style:

```python
conservative_prompt = """
You are a conservative prediction market trader.
Prioritize risk management and require high conviction for trades.
Always consider worst-case scenarios...
"""

agent = create_deep_agent(
    tools=tools,
    system_prompt=conservative_prompt
)
```

## 🔄 Integration Patterns

### Hybrid Approach (Recommended)

Use deep agent for research, traditional agent for execution:

```python
# Step 1: Deep research for comprehensive analysis
research = analyze_market_with_deep_research(market_question)

# Step 2: Feed insights to traditional agent for structured decision
decision = analyze_bet(market_question, market_id)

# Step 3: Combine insights for final recommendation
final_rec = combine_analyses(research, decision)
```

### Conditional Usage

Use deep agent only for complex markets:

```python
def smart_analyze(question):
    complexity = assess_complexity(question)

    if complexity > 0.7:  # Complex market
        return analyze_market_with_deep_research(question)
    else:  # Simple market
        return analyze_bet(question)
```

## 📊 Performance Considerations

### Cost Optimization

- **Batch Research**: Use comprehensive_research tool for multiple queries
- **Cached Results**: File system tools enable result caching
- **Selective Depth**: Adjust web search result limits based on needs

### Speed vs Depth Trade-offs

- **Fast Mode**: Limit web searches, focus on market data
- **Deep Mode**: Extensive research with multiple sources
- **Balanced Mode**: Targeted research with key insights

## 🧪 Testing & Validation

Run the demo to validate integration:

```bash
cd /home/orson-dev/projects/polymarket_langchain
python demo_deep_agent.py
```

Expected outputs:
- Comprehensive market analysis reports
- Web search integration working
- File system context management
- Subagent task delegation

## 🚨 Best Practices

### Research Quality
- Always cross-reference multiple sources
- Consider publication dates and credibility
- Look for consensus vs. outlier opinions

### Risk Management
- Use deep research for conviction, not gambling
- Validate web research with market data
- Maintain position size limits regardless of research depth

### System Monitoring
- Track API usage costs
- Monitor research quality vs. trading performance
- Log agent decisions for continuous improvement

## 🔮 Future Enhancements

Potential additions to the deep agent system:

1. **Sentiment Analysis**: Integrate news sentiment scoring
2. **Correlation Analysis**: Multi-market relationship detection
3. **Expert Networks**: Query domain expert opinions
4. **Real-time Monitoring**: Continuous market intelligence updates
5. **Portfolio Optimization**: Multi-position risk management

## 📞 Support & Resources

- **DeepAgents Docs**: https://docs.langchain.com/oss/deepagents/
- **Tavily Search**: https://tavily.com
- **Polymarket API**: Your existing Gamma client integration

The deep research agent enhances your systematic trading approach while maintaining the rigorous analysis framework you've built. Use it strategically for complex analysis where web-scale research provides clear edge opportunities.

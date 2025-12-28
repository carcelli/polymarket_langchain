# LangSmith Deployment Guide for DeepAgents Polymarket System

This guide explains how to deploy the comprehensive deepagents-based Polymarket trading system using LangSmith Deployment.

## 📋 Application Structure

Our application follows the LangSmith deployment structure:

```
polymarket_langchain/
├── agents/                          # Core agent implementations
│   ├── deep_research_agent.py      # DeepAgents implementation
│   ├── tools/                      # Research and trading tools
│   └── graph/                      # Original LangGraph agents
├── langgraph.json                  # Deployment configuration
├── requirements.txt                # Python dependencies
├── .env                            # Environment variables
├── .env.example                    # Environment template
└── DEEP_AGENT_README.md            # Implementation documentation
```

## 🔧 Configuration File (`langgraph.json`)

```json
{
  "$schema": "https://langchain-ai.github.io/langgraph/schemas/langgraph.json",
  "graphs": {
    "deep_research_agent": "./agents/deep_research_agent.py:create_polymarket_research_agent",
    "memory_enabled_agent": "./agents/deep_research_agent.py:create_memory_enabled_agent",
    "trading_agent": "./agents/deep_research_agent.py:create_trading_agent_with_approval",
    "conservative_agent": "./agents/deep_research_agent.py:create_conservative_trading_agent",
    "autonomous_agent": "./agents/deep_research_agent.py:create_autonomous_research_agent"
  },
  "env": ".env",
  "dependencies": [
    "deepagents",
    "langchain-openai",
    "langchain-anthropic",
    "tavily-python",
    "py-clob-client",
    "python-dotenv",
    "langgraph",
    "langsmith",
    "pydantic",
    "."
  ]
}
```

## 🚀 Available Graphs for Deployment

### Primary DeepAgents Graphs

| Graph Name | Function | Description |
|------------|----------|-------------|
| `deep_research_agent` | `create_polymarket_research_agent()` | Full-featured research agent with subagents, memory, and HITL |
| `memory_enabled_agent` | `create_memory_enabled_agent()` | Agent with long-term memory capabilities |
| `trading_agent` | `create_trading_agent_with_approval()` | Trading agent with human-in-the-loop approvals |

### Specialized Configurations

| Graph Name | Function | Description |
|------------|----------|-------------|
| `conservative_agent` | `create_conservative_trading_agent()` | Conservative trading with strict risk controls |
| `autonomous_agent` | `create_autonomous_research_agent()` | Pure research agent without trading capabilities |

## 📦 Dependencies

### Core Dependencies
- **deepagents**: Main framework providing middleware capabilities
- **langchain-openai/anthropic**: LLM integrations
- **tavily-python**: Web search capabilities
- **py-clob-client**: Polymarket API integration
- **langgraph**: Graph orchestration
- **langsmith**: Deployment and observability

### Local Dependencies
- **`.`**: Current package with all agent implementations

## 🔐 Environment Variables

Required environment variables in `.env`:

```bash
# LLM APIs
ANTHROPIC_API_KEY="your-anthropic-key"
OPENAI_API_KEY="your-openai-key"

# Research Tools
TAVILY_API_KEY="your-tavily-key"

# Trading (Optional)
POLYGON_WALLET_PRIVATE_KEY="your-wallet-key"

# Observability
LANGSMITH_API_KEY="your-langsmith-key"
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=polymarket-agent
```

## 🛠️ Deployment Steps

### 1. Install LangSmith CLI
```bash
pip install langsmith
```

### 2. Authenticate
```bash
langsmith auth login
```

### 3. Deploy Application
```bash
# Deploy all graphs
langsmith deploy

# Or deploy specific graph
langsmith deploy --graph deep_research_agent
```

### 4. Check Deployment Status
```bash
langsmith deploy status
```

## 🎯 Graph Usage Examples

### Basic Research Agent
```python
from langsmith import Client

client = Client()
agent = client.remote_runnable("deep_research_agent")

result = agent.invoke({
    "messages": [{"role": "user", "content": "Analyze Bitcoin market trends"}]
})
```

### Memory-Enabled Agent
```python
agent = client.remote_runnable("memory_enabled_agent")

# First interaction
result1 = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer conservative trading strategies"}]
})

# Agent remembers preferences in subsequent interactions
result2 = agent.invoke({
    "messages": [{"role": "user", "content": "What are my trading preferences?"}]
})
```

### Trading Agent with Approvals
```python
agent = client.remote_runnable("trading_agent")

# Agent will request human approval for trades
result = agent.invoke({
    "messages": [{"role": "user", "content": "Execute a YES trade on this market"}]
})
# Returns interrupt information for human approval workflow
```

## 🔍 Middleware Architecture in Deployment

Our deepagents implementation leverages the full middleware stack:

### TodoListMiddleware
- **Provides**: `write_todos` tool for task planning
- **Deployment Benefit**: Structured research workflows in production

### FilesystemMiddleware
- **Provides**: File operations (ls, read_file, write_file, edit_file)
- **Deployment Benefit**: Context management and large result handling

### SubAgentMiddleware
- **Provides**: `task` tool for subagent delegation
- **Deployment Benefit**: Context isolation and specialized processing

## 🏗️ Advanced Deployment Patterns

### Multiple Graph Deployment
Deploy different agent configurations for different use cases:

```json
{
  "graphs": {
    "research_only": "./agents/deep_research_agent.py:create_autonomous_research_agent",
    "conservative_trading": "./agents/deep_research_agent.py:create_conservative_trading_agent",
    "full_featured": "./agents/deep_research_agent.py:create_polymarket_research_agent"
  }
}
```

### Environment-Specific Configurations
Use different configurations for development/production:

```json
{
  "graphs": {
    "dev_agent": "./agents/deep_research_agent.py:create_polymarket_research_agent",
    "prod_agent": "./agents/deep_research_agent.py:create_trading_agent_with_approval"
  }
}
```

## 📊 Monitoring and Observability

### LangSmith Tracing
All agents include automatic tracing:
```python
# Traces are automatically captured
result = agent.invoke(input)
# View in LangSmith dashboard
```

### Custom Metrics
Monitor agent performance:
- Subagent delegation patterns
- Interrupt approval rates
- Memory utilization
- Research completion times

## 🚨 Production Considerations

### Security
- Store API keys securely in environment variables
- Use human-in-the-loop for trading operations
- Implement rate limiting for external APIs

### Scalability
- Deploy multiple graph instances for different workloads
- Use appropriate memory backends (PostgresStore for production)
- Monitor resource usage and optimize as needed

### Reliability
- Implement proper error handling in nodes
- Use checkpointers for state persistence
- Monitor agent performance and success rates

## 🔄 CI/CD Integration

### Automated Testing
```bash
# Test all graphs before deployment
python -m pytest tests/
```

### Deployment Pipeline
```yaml
# .github/workflows/deploy.yml
name: Deploy to LangSmith
on: [push]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install langsmith
      - run: langsmith auth login --api-key ${{ secrets.LANGSMITH_API_KEY }}
      - run: langsmith deploy
```

## 📚 Additional Resources

- [LangSmith Deployment Documentation](https://docs.smith.langchain.com/)
- [DeepAgents Framework](https://docs.langchain.com/oss/deepagents/)
- [LangGraph Deployment Guide](https://langchain-ai.github.io/langgraph/tutorials/deployment/)

## 🎯 Deployment Checklist

- ✅ `langgraph.json` configured with all graphs
- ✅ `requirements.txt` includes all dependencies
- ✅ `.env` contains all required API keys
- ✅ Environment variables match deployment requirements
- ✅ All graphs tested locally before deployment
- ✅ Monitoring and tracing configured
- ✅ Security policies implemented (HITL for trading)
- ✅ Backup and recovery procedures documented

---

**Ready for production deployment with enterprise-grade AI agent capabilities!** 🚀

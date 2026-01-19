# Polymarket LangChain Agents Implementation

## Overview

This implementation provides LangChain agents that interact with Polymarket prediction markets, with a focus on **crypto and sports markets**. The agents enable probability extraction, business forecasting, and ML model validation for businesses in these high-growth sectors.

## Key Features

### ✅ **Probability Extraction Agent**
- Extracts implied probabilities from Polymarket prices
- Focuses on business-relevant events (recessions, elections, economic indicators)
- Converts market prices to actionable probability estimates

### ✅ **ML Forecast Comparison Agent**
- Compares your internal ML model predictions against market consensus
- Identifies when your models outperform or underperform crowd wisdom
- Provides business insights on forecast reliability

### ✅ **Business Domain Agents**
- **Crypto** ⭐ *Primary Focus*: Bitcoin/Ethereum prices, adoption rates, regulation, DeFi events
- **Sports** ⭐ *Primary Focus*: Championships, tournaments, player performance, betting markets
- **Economy**: Recession risks, Fed policy, GDP forecasts
- **Politics**: Election outcomes, policy changes
- Specialized filtering and analysis for each domain

## Quick Start

### 1. Installation

```bash
cd /home/orson-dev/projects/polymarket_langchain
pip install -e .
```

### 2. Environment Setup

```bash
export OPENAI_API_KEY="your-openai-api-key-here"
# Optional: Set MARKET_FOCUS for category filtering (crypto/sports)
export MARKET_FOCUS="crypto"
```

### 3. Basic Usage

```python
from polymarket_agents.langchain.agent import (
    create_crypto_agent,
    create_sports_agent,
    compare_ml_vs_market_forecast,
    analyze_business_risks
)

# Crypto-focused agent for market analysis
crypto_agent = create_crypto_agent()
result = crypto_agent.run({
    "input": "What are current Polymarket odds on Bitcoin reaching $200k in 2026?"
})

# Sports agent for championship predictions
sports_agent = create_sports_agent()
result = sports_agent.run({
    "input": "What are the odds for different teams winning the Super Bowl?"
})

# Compare your ML forecast against market
comparison = compare_ml_vs_market_forecast(
    ml_forecast=0.65,  # Your model's 65% Bitcoin prediction
    event_description="Bitcoin price above $100k by end of 2026"
)

# Analyze crypto business risks (default domain is now crypto)
risks = analyze_business_risks(
    business_type="crypto trading firm"
)
```

## Agent Types & Use Cases

### Probability Extraction Agent
**Best for**: Business forecasting and risk assessment
```python
agent = create_probability_extraction_agent()
query = """
Extract implied probabilities for major 2026 economic events.
Focus on recession risks, Fed policy, and GDP growth.
Summarize top 5 markets with highest business impact.
"""
```

### ML Forecast Comparison Agent
**Best for**: Model validation and calibration
```python
# Your ML model predicts 42% chance of event X
comparison = compare_ml_vs_market_forecast(0.42, "Event X description")
# Returns analysis of model vs market differences
```

### Business Domain Agents
**Best for**: Industry-specific risk analysis
```python
# Crypto-focused agent for blockchain business planning
crypto_agent = create_crypto_agent()
result = crypto_agent.run({"input": "What crypto market opportunities exist?"})

# Sports agent for betting business analysis
sports_agent = create_sports_agent()
result = sports_agent.run({"input": "What are high-value sports markets?"})

# Direct business risk analysis (defaults to crypto)
risks = analyze_business_risks("crypto trading firm")
```

## Target Events for Testing

### High-Impact Crypto & Sports Events ⭐
- **Bitcoin Price Targets**: $100k, $200k, $500k predictions for 2026
- **Ethereum ETF Approval**: SEC approval probabilities and timing
- **Super Bowl Winner**: NFL championship outcome predictions
- **World Series Champion**: MLB season predictions
- **NBA Finals**: Basketball championship odds
- **Crypto Regulation**: Major regulatory decision impacts
- **Champions League**: Soccer tournament outcomes
- **March Madness**: College basketball tournament predictions

### Business Applications
- **Crypto Trading**: Price movement predictions and risk assessment
- **Sports Betting**: Championship odds and betting market analysis
- **Investment Decisions**: Market sentiment validation for crypto assets
- **Regulatory Risk**: Compliance planning for crypto businesses
- **Tournament Planning**: Sports event outcome probabilities

## Architecture

### Existing Tools (Already Implemented)
- `get_current_markets_gamma()`: Fetch current active markets
- `get_superforecast()`: Structured forecasting methodology
- `search_markets_db()`: Database search capabilities
- `analyze_market_with_llm()`: General market analysis
- `get_top_volume_markets()`: High-volume market discovery

### New Agents Built On Top
- **ProbabilityExtractionAgent**: Focused on probability extraction
- **MLComparisonAgent**: Model vs market validation
- **BusinessDomainAgent**: Industry-specific analysis
- **ResearchAgent**: Multi-source information gathering

## Testing

Run the test suite to validate functionality:

```bash
python test_polymarket_agents.py
```

This tests:
- Agent creation and initialization
- Basic query execution (requires OpenAI API key)
- Error handling and edge cases

## Business Value

### For Small Business Owners
1. **Risk Assessment**: Quantify business risks using crowd wisdom
2. **Forecast Validation**: Compare internal predictions against market consensus
3. **Strategic Planning**: Make data-driven decisions on expansion, hiring, inventory
4. **Competitive Intelligence**: Monitor market sentiment on industry events

### For ML Engineers
1. **Model Calibration**: Use market data to improve forecast accuracy
2. **Bias Detection**: Identify when models diverge from crowd wisdom
3. **Feature Engineering**: Incorporate market probabilities as features
4. **Backtesting**: Historical market data for model validation

## Next Steps

### Immediate Priorities
1. **Test with real data**: Run agents on live Polymarket data
2. **Customize prompts**: Adapt agent behavior for your specific business
3. **Integrate ML pipeline**: Connect to your existing forecasting models
4. **Add monitoring**: Set up alerts for key market movements

### Future Enhancements
- **Historical backtesting**: Integrate with existing NumPy evaluation metrics
- **Multi-agent workflows**: LangGraph for complex analysis chains
- **Real-time monitoring**: Automated market scanning and alerts
- **Portfolio optimization**: Trading strategy recommendations

## Technical Notes

### Dependencies
- LangChain for agent framework
- OpenAI API for LLM capabilities
- Polymarket Gamma API (no authentication required)
- Existing Polymarket connectors and tools

### Performance
- Agent latency: ~5-10 seconds per query
- API rate limits: Gamma API is generous for reads
- Cost: ~$0.01-0.05 per agent interaction

### Security
- Read-only operations (no trading capabilities enabled)
- No private key exposure
- Safe for production business use

## Troubleshooting

### Common Issues
1. **Import errors**: Run `pip install -e .` from project root
2. **API key missing**: Set `OPENAI_API_KEY` environment variable
3. **Network timeouts**: Gamma API calls may timeout; retry logic included
4. **Empty results**: Some markets may not have data; try different queries

### Getting Help
- Check existing tool documentation in `tools.py`
- Review agent examples in `agent.py`
- Run test suite for validation: `python test_polymarket_agents.py`

---

**Ready to get started?** Run `python test_polymarket_agents.py` to validate your setup, then try the probability extraction agent on current economic markets!
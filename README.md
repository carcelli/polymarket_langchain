# Polymarket LangChain Agent

This project recreates the core components from the [Polymarket Agents](https://github.com/polymarket/agents) repository, integrated with LangChain for building AI-powered trading agents on Polymarket.

## What's Included

This migration includes all the relevant components from the original agents project:

### Core Components
- **Polymarket API Integration** (`agents/polymarket/`)
  - `polymarket.py` - Main Polymarket API client with trading functionality
  - `gamma.py` - Gamma API client for market and event data

### Data Models & Utils
- **Data Models** (`agents/utils/objects.py`) - Pydantic models for markets, events, trades
- **Utilities** (`agents/utils/utils.py`) - Helper functions

### Data Connectors
- **ChromaDB** (`agents/connectors/chroma.py`) - Vector database for RAG
- **News API** (`agents/connectors/news.py`) - News aggregation
- **Web Search** (`agents/connectors/search.py`) - Web search integration

### Application Logic
- **Trading** (`agents/application/trade.py`) - Main trading functionality
- **Executor** (`agents/application/executor.py`) - AI-powered trade execution
- **Creator** (`agents/application/creator.py`) - Market creation utilities
- **Prompts** (`agents/application/prompts.py`) - AI prompts and templates

### Scripts
- **CLI** (`scripts/python/cli.py`) - Command-line interface for all operations
- **Server** (`scripts/python/server.py`) - FastAPI server for remote access
- **Setup** (`scripts/python/setup.py`) - Setup utilities

## Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate polymarket-agent
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Install additional dependencies if needed:
```bash
pip install -r requirements.txt
```

## Usage

### CLI Commands

Get markets:
```bash
python scripts/python/cli.py get-all-markets --limit 5
```

Get news:
```bash
python scripts/python/cli.py get-relevant-news "market keywords"
```

Query local RAG:
```bash
python scripts/python/cli.py create-local-markets-rag ./data
python scripts/python/cli.py query-local-markets-rag ./data "query"
```

### Trading

Execute trades:
```bash
python agents/application/trade.py
```

## Environment Variables

Required:
- `POLYGON_WALLET_PRIVATE_KEY` - Your Polygon wallet private key
- `OPENAI_API_KEY` - OpenAI API key for AI functionality

Optional:
- `NEWS_API_KEY` - For news aggregation
- Other API keys as needed

## Architecture

The project maintains the same modular architecture as the original:

- **APIs**: Standardized data sources and order types
- **Connectors**: External data integrations (ChromaDB, NewsAPI, Search)
- **Application**: Core trading and AI logic
- **Utils**: Data models and utilities

## Dependencies

All dependencies from the original project are included in both `environment.yml` (for conda) and `requirements.txt` (for pip).

## Original Source

This project is based on the [Polymarket Agents](https://github.com/polymarket/agents) repository, adapted for integration with LangChain workflows.

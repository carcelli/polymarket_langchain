"""
Configuration module for Polymarket LangGraph agents.

Centralized configuration management for environment variables and settings.
"""

import os
from typing import Optional

# =============================================================================
# MARKET FOCUS CONFIGURATION
# =============================================================================

# Focus category for market filtering (e.g., "sports", "politics", None for all)
# Set via MARKET_FOCUS environment variable
MARKET_FOCUS: Optional[str] = os.getenv("MARKET_FOCUS")

# =============================================================================
# API CONFIGURATION
# =============================================================================

OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY: Optional[str] = os.getenv("TAVILY_API_KEY")
NEWSAPI_API_KEY: Optional[str] = os.getenv("NEWSAPI_API_KEY")
GITHUB_APP_ID: Optional[str] = os.getenv("GITHUB_APP_ID")
GITHUB_APP_PRIVATE_KEY: Optional[str] = os.getenv("GITHUB_APP_PRIVATE_KEY")
GITHUB_REPOSITORY: Optional[str] = os.getenv("GITHUB_REPOSITORY")

# Polymarket/CLOB configuration
POLYGON_WALLET_PRIVATE_KEY: Optional[str] = os.getenv("POLYGON_WALLET_PRIVATE_KEY")
CLOB_API_KEY: Optional[str] = os.getenv("CLOB_API_KEY")
CLOB_SECRET: Optional[str] = os.getenv("CLOB_SECRET")
CLOB_PASS_PHRASE: Optional[str] = os.getenv("CLOB_PASS_PHRASE")
CLOB_API_URL: str = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
CHAIN_ID: int = int(os.getenv("CHAIN_ID", "137"))  # Polygon mainnet

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

DATABASE_PATH: str = os.getenv("DATABASE_PATH", "data/markets.db")

# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================

DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"

# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate critical configuration settings."""
    missing = []

    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")

    if MARKET_FOCUS and MARKET_FOCUS not in ["sports", "politics", "crypto", "tech", "geopolitics", "culture", "finance", "economy", "science"]:
        print(f"WARNING: MARKET_FOCUS='{MARKET_FOCUS}' is not a standard category. This may filter out all markets.")

    if missing:
        print(f"WARNING: Missing required environment variables: {', '.join(missing)}")

# Run validation on import
validate_config()
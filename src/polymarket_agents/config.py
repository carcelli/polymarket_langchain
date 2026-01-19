import os
from dotenv import load_dotenv

load_dotenv()

def get_polygon_private_key() -> str:
    """Load wallet private key: raw string > file path > error."""
    raw = os.getenv("POLYGON_WALLET_PRIVATE_KEY")
    if raw and raw.strip():
        return raw.strip()

    path = os.getenv("POLYGON_WALLET_KEY_FILE")
    if path and os.path.exists(path):
        with open(path, "r") as f:
            content = f.read().strip()
            if content.startswith("-----BEGIN"):
                return content
            raise ValueError(f"File {path} does not contain valid PEM content")

    raise ValueError(
        "No Polygon private key found. Set POLYGON_WALLET_PRIVATE_KEY or "
        "POLYGON_WALLET_KEY_FILE in .env"
    )

# Existing keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
GITHUB_APP_ID = os.getenv("GITHUB_APP_ID")
GITHUB_APP_PRIVATE_KEY = os.getenv("GITHUB_APP_PRIVATE_KEY")
GITHUB_REPOSITORY = os.getenv("GITHUB_REPOSITORY")

# Polymarket/CLOB configuration
CLOB_API_KEY = os.getenv("CLOB_API_KEY")
CLOB_SECRET = os.getenv("CLOB_SECRET")
CLOB_PASS_PHRASE = os.getenv("CLOB_PASS_PHRASE")
CLOB_API_URL: str = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
CHAIN_ID: int = int(os.getenv("CHAIN_ID", "137"))  # Polygon mainnet

# =============================================================================
# MARKET FOCUS CONFIGURATION
# =============================================================================

# Focus category for market filtering (e.g., "sports", "politics", None for all)
# Set via MARKET_FOCUS environment variable
MARKET_FOCUS = os.getenv("MARKET_FOCUS")

# =============================================================================
# LLM CONFIGURATION
# =============================================================================

# Default model for LangChain agents
# Can be overridden per agent call, but provides centralized control
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

# Default temperature for different agent types
DEFAULT_TEMPERATURE_ANALYSIS = 0.1  # For factual analysis
DEFAULT_TEMPERATURE_CREATIVE = 0.3  # For research/generation
DEFAULT_TEMPERATURE_STRUCTURED = 0.0  # For structured outputs

# Model configurations for different use cases
STRUCTURED_OUTPUT_MODELS = ["gpt-4o-2024-08-06", "gpt-4o-mini"]  # Models that support structured output well
HIGH_REASONING_MODELS = ["gpt-4o-2024-08-06", "gpt-4o"]  # For complex analysis

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
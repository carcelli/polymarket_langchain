import json
import time
from datetime import datetime
from typing import List, Dict, Any

from polymarket_agents.langchain.tools import (
    fetch_tradeable_markets,
    search_news,
    get_market_by_token,
)
from polymarket_agents.memory.manager import MemoryManager
from polymarket_agents.utils.structures import StrKeyDict


class IngestionTeam:
    """
    A team of agents (simulated by tool calls) that ingests Polymarket data.

    Roles:
    - Scout: Finds active markets.
    - Researcher: Finds relevant news.
    - Archivist: Stores the data.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.memory = MemoryManager(db_path=f"{self.data_dir}/memory.db")

        # Robust Cache for Ingestion
        # This prevents duplicate work even if IDs come in as ints or strings
        self.processed_markets = StrKeyDict()

    def run_cycle(self, limit: int = 5):
        """Runs one ingestion cycle."""
        print(f"[{datetime.now()}] Starting ingestion cycle...")

        # 1. Scout: Fetch markets
        print("  - Scout: Fetching active markets...")
        try:
            markets_json = fetch_tradeable_markets.invoke({"limit": limit})
            markets = json.loads(markets_json)
        except Exception as e:
            print(f"  ! Scout failed: {e}")
            return

        # 2. Researcher & Archivist Loop
        for market in markets:
            market_id = market.get("id")

            # Check cache using the robust dict
            if market_id in self.processed_markets:
                print(f"Skipping known market {market_id}")
                continue

            question = market.get("question")
            print(f"  - Researcher: Analyzing market '{question}' ({market_id})...")

            # Search news
            print(f"    - Searching news for: {question}")
            try:
                # Use the question as keywords
                news_json = search_news.invoke({"keywords": question})
                if news_json.strip().startswith("Error"):
                    print(f"    ! News tool returned error: {news_json}")
                    news_articles = []
                else:
                    news_articles = json.loads(news_json)
            except json.JSONDecodeError:
                print(f"    ! Failed to parse news JSON. Raw output: {news_json}")
                news_articles = []
            except Exception as e:
                print(f"    ! News search failed: {e}")
                news_articles = []

            # 3. Archivist: Save to Memory System
            print(f"  - Archivist: Storing market {market_id} in memory...")
            try:
                self.memory.add_market(market, news_articles)

                # Store in cache to prevent duplicate processing
                self.processed_markets[market_id] = True

            except Exception as e:
                print(f"  ! Archivist failed to save to DB: {e}")

            # Be polite to APIs
            time.sleep(0.5)

        print("  - Ingestion complete.")


if __name__ == "__main__":
    team = IngestionTeam()
    team.run_cycle(limit=3)

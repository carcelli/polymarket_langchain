"""
Connector for Polymarket 15-minute Bitcoin "Up or Down" markets.

These markets resolve based on Chainlink BTC/USD price feed every 15 minutes.
"""

import requests
import json
from typing import Dict, Optional, List
from datetime import datetime, timezone, timedelta
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


class UpDownMarketConnector:
    """Fetch and track 15-minute Bitcoin Up or Down markets on Polymarket."""

    GAMMA_API_BASE = "https://gamma-api.polymarket.com"
    CLOB_API_BASE = "https://clob.polymarket.com"
    WEB_BASE = "https://polymarket.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        )

    def get_current_15min_slot(self) -> datetime:
        """Get the current 15-minute time slot (rounded down)."""
        now = datetime.now(timezone.utc)
        minute = (now.minute // 15) * 15
        return now.replace(minute=minute, second=0, microsecond=0)

    def get_next_15min_slot(self) -> datetime:
        """Get the next 15-minute time slot."""
        current_slot = self.get_current_15min_slot()
        return current_slot + timedelta(minutes=15)

    def generate_market_url(self, timestamp: int) -> str:
        """Generate Polymarket URL for a 15-min market at given timestamp."""
        return f"{self.WEB_BASE}/event/btc-updown-15m-{timestamp}"

    def scrape_market_id(self, timestamp: int) -> Optional[str]:
        """
        Scrape market ID from Polymarket web page.

        Args:
            timestamp: Unix timestamp for the market start time

        Returns:
            Market ID string or None if not found
        """
        url = self.generate_market_url(timestamp)

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Extract Next.js data
            next_data_script = soup.find("script", id="__NEXT_DATA__")
            if not next_data_script:
                logger.warning(f"No __NEXT_DATA__ found at {url}")
                return None

            data = json.loads(next_data_script.string)
            page_props = data.get("props", {}).get("pageProps", {})

            # Navigate to market data
            dehydrated_state = page_props.get("dehydratedState", {})
            queries = dehydrated_state.get("queries", [])

            for query in queries:
                state = query.get("state", {})
                query_data = state.get("data", {})

                # Look for markets array
                if "markets" in query_data:
                    markets = query_data["markets"]
                    if markets and isinstance(markets, list):
                        market = markets[0]
                        market_id = market.get("id")
                        if market_id:
                            logger.info(f"Found market ID {market_id} at {url}")
                            return str(market_id)

            logger.warning(f"No market ID found in page data at {url}")
            return None

        except Exception as e:
            logger.error(f"Error scraping market ID from {url}: {e}")
            return None

    def get_current_market_id(self) -> Optional[str]:
        """Get the market ID for the current 15-minute slot."""
        current_slot = self.get_current_15min_slot()
        timestamp = int(current_slot.timestamp())
        return self.scrape_market_id(timestamp)

    def get_next_market_id(self) -> Optional[str]:
        """Get the market ID for the next 15-minute slot."""
        next_slot = self.get_next_15min_slot()
        timestamp = int(next_slot.timestamp())
        return self.scrape_market_id(timestamp)

    def fetch_market_data(self, market_id: str) -> Optional[Dict]:
        """
        Fetch market data from Gamma API.

        Args:
            market_id: Market ID string

        Returns:
            Dictionary with market data or None if error
        """
        url = f"{self.GAMMA_API_BASE}/markets/{market_id}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching market {market_id} from Gamma API: {e}")
            return None

    def get_current_market_data(self) -> Optional[Dict]:
        """
        Get complete data for the current 15-minute market.

        Returns:
            Dictionary with market data including:
            - market_id: str
            - question: str
            - volume: float
            - liquidity: float
            - up_price: float (probability of Up outcome)
            - down_price: float (probability of Down outcome)
            - active: bool
            - end_date: str (ISO format)
            - condition_id: str (for CLOB API)
        """
        market_id = self.get_current_market_id()
        if not market_id:
            logger.warning("Could not find current market ID")
            return None

        market_data = self.fetch_market_data(market_id)
        if not market_data:
            return None

        # Extract and normalize data
        try:
            # Handle different outcome price formats
            # API returns outcomePrices as a JSON string: "[\"0.485\", \"0.515\"]"
            outcome_prices_raw = market_data.get("outcomePrices", '["0.5", "0.5"]')

            # Parse prices
            up_price = 0.5
            down_price = 0.5

            if outcome_prices_raw:
                try:
                    if isinstance(outcome_prices_raw, str):
                        # Parse JSON string
                        outcome_prices = json.loads(outcome_prices_raw)
                    else:
                        outcome_prices = outcome_prices_raw

                    if isinstance(outcome_prices, list) and len(outcome_prices) >= 2:
                        up_price = float(outcome_prices[0])
                        down_price = float(outcome_prices[1])
                except (ValueError, TypeError, json.JSONDecodeError) as e:
                    logger.warning(
                        f"Could not parse outcome prices: {outcome_prices_raw}, error: {e}"
                    )

            # Parse volume and liquidity
            volume = market_data.get("volume", 0)
            if isinstance(volume, str):
                volume = float(volume) if volume else 0

            liquidity = market_data.get("liquidity", 0)
            if isinstance(liquidity, str):
                liquidity = float(liquidity) if liquidity else 0

            return {
                "market_id": market_id,
                "question": market_data.get("question", ""),
                "volume": float(volume),
                "liquidity": float(liquidity),
                "up_price": up_price,
                "down_price": down_price,
                "active": market_data.get("active", False),
                "closed": market_data.get("closed", False),
                "end_date": market_data.get("endDate", ""),
                "condition_id": market_data.get("conditionId", ""),
                "slug": market_data.get("slug", ""),
                "created_at": market_data.get("createdAt", ""),
                "updated_at": market_data.get("updatedAt", ""),
            }
        except Exception as e:
            logger.error(f"Error parsing market data: {e}")
            logger.debug(f"Market data: {market_data}")
            return None

    def get_upcoming_markets(self, num_intervals: int = 5) -> List[Dict]:
        """
        Get market IDs for upcoming 15-minute slots.

        Args:
            num_intervals: Number of future intervals to fetch

        Returns:
            List of dictionaries with timestamp and URL for each interval
        """
        current_slot = self.get_current_15min_slot()
        markets = []

        for i in range(num_intervals):
            slot_time = current_slot + timedelta(minutes=15 * i)
            timestamp = int(slot_time.timestamp())

            markets.append(
                {
                    "timestamp": timestamp,
                    "datetime": slot_time.isoformat(),
                    "url": self.generate_market_url(timestamp),
                    "market_id": None,  # Will be populated when scraped
                }
            )

        return markets

    def scrape_all_upcoming_markets(self, num_intervals: int = 5) -> List[Dict]:
        """
        Scrape market IDs for all upcoming intervals.

        This is useful for pre-loading market IDs before they become active.
        """
        upcoming = self.get_upcoming_markets(num_intervals)

        for market in upcoming:
            market_id = self.scrape_market_id(market["timestamp"])
            market["market_id"] = market_id

            if market_id:
                # Fetch full data
                data = self.fetch_market_data(market_id)
                if data:
                    market["volume"] = float(data.get("volume", 0))
                    market["active"] = data.get("active", False)

        return upcoming


# Convenience functions
def get_current_15min_market() -> Optional[Dict]:
    """Quick function to get current market data."""
    connector = UpDownMarketConnector()
    return connector.get_current_market_data()


def get_current_market_id() -> Optional[str]:
    """Quick function to get current market ID."""
    connector = UpDownMarketConnector()
    return connector.get_current_market_id()


if __name__ == "__main__":
    # Test the connector
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 70)
    print("15-MINUTE BITCOIN MARKET CONNECTOR TEST")
    print("=" * 70 + "\n")

    connector = UpDownMarketConnector()

    # Get current slot info
    current_slot = connector.get_current_15min_slot()
    print(f"Current 15-min slot: {current_slot}")
    print(f"Timestamp: {int(current_slot.timestamp())}\n")

    # Get current market data
    print("Fetching current market data...")
    market_data = connector.get_current_market_data()

    if market_data:
        print(f"\n✅ Current Market Found:")
        print(f"   ID: {market_data['market_id']}")
        print(f"   Question: {market_data['question']}")
        print(f"   Volume: ${market_data['volume']:,.2f}")
        print(f"   Liquidity: ${market_data['liquidity']:,.2f}")
        print(
            f"   Up Price: {market_data['up_price']:.3f} ({market_data['up_price']*100:.1f}%)"
        )
        print(
            f"   Down Price: {market_data['down_price']:.3f} ({market_data['down_price']*100:.1f}%)"
        )
        print(f"   Active: {market_data['active']}")
        print(f"   End Date: {market_data['end_date']}")
    else:
        print("❌ Could not fetch current market data")

    # Show upcoming markets
    print(f"\n{'='*70}")
    print("UPCOMING MARKETS")
    print("=" * 70 + "\n")

    upcoming = connector.get_upcoming_markets(3)
    for i, market in enumerate(upcoming, 1):
        print(f"{i}. {market['datetime']}")
        print(f"   URL: {market['url']}")

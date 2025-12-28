"""
Polymarket Agents Test Suite

Tests to verify correct setup of polymarket agent utilities:
- Module imports
- Data models (Pydantic objects)
- API clients (Gamma, Polymarket)
- Connectors (News, Search, Chroma)
- Application logic (Executor, Trader, Creator, Prompts)
- CLI commands

Run with:
    python -m pytest tests/ -v
    # or
    python -m unittest discover tests/
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock
from datetime import datetime

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestProjectStructure(unittest.TestCase):
    """Verify project file structure matches expected layout."""

    def test_agents_application_modules_exist(self):
        """Verify all application modules exist."""
        from agents.application import cron, creator, trade, executor, prompts
        
        self.assertTrue(hasattr(cron, 'Scheduler'))
        self.assertTrue(hasattr(cron, 'TradingAgent'))
        self.assertTrue(hasattr(creator, 'Creator'))
        self.assertTrue(hasattr(trade, 'Trader'))
        self.assertTrue(hasattr(executor, 'Executor'))
        self.assertTrue(hasattr(prompts, 'Prompter'))

    def test_agents_connectors_modules_exist(self):
        """Verify all connector modules exist."""
        from agents.connectors import news, search, chroma
        
        self.assertTrue(hasattr(news, 'News'))
        self.assertTrue(hasattr(chroma, 'PolymarketRAG'))

    def test_agents_polymarket_modules_exist(self):
        """Verify polymarket modules exist."""
        from agents.polymarket import gamma, polymarket
        
        self.assertTrue(hasattr(gamma, 'GammaMarketClient'))
        self.assertTrue(hasattr(polymarket, 'Polymarket'))

    def test_agents_utils_modules_exist(self):
        """Verify utils modules exist."""
        from agents.utils import objects, utils
        
        self.assertTrue(hasattr(objects, 'Trade'))
        self.assertTrue(hasattr(objects, 'SimpleMarket'))
        self.assertTrue(hasattr(objects, 'SimpleEvent'))
        self.assertTrue(hasattr(objects, 'Market'))
        self.assertTrue(hasattr(objects, 'PolymarketEvent'))


class TestDataModels(unittest.TestCase):
    """Test Pydantic data models for market/event representations."""

    def test_simple_market_model(self):
        """Test SimpleMarket model validation."""
        from agents.utils.objects import SimpleMarket
        
        market_data = {
            "id": 123,
            "question": "Will X happen?",
            "end": "2025-12-31",
            "description": "Test market description",
            "active": True,
            "funded": True,
            "rewardsMinSize": 10.0,
            "rewardsMaxSpread": 0.05,
            "spread": 0.02,
            "outcomes": '["Yes", "No"]',
            "outcome_prices": '["0.5", "0.5"]',
            "clob_token_ids": '["token1", "token2"]',
        }
        market = SimpleMarket(**market_data)
        
        self.assertEqual(market.id, 123)
        self.assertEqual(market.question, "Will X happen?")
        self.assertTrue(market.active)

    def test_simple_event_model(self):
        """Test SimpleEvent model validation."""
        from agents.utils.objects import SimpleEvent
        
        event_data = {
            "id": 456,
            "ticker": "TEST",
            "slug": "test-event",
            "title": "Test Event",
            "description": "Test event description",
            "end": "2025-12-31",
            "active": True,
            "closed": False,
            "archived": False,
            "restricted": False,
            "new": True,
            "featured": False,
            "markets": "123,456",
        }
        event = SimpleEvent(**event_data)
        
        self.assertEqual(event.id, 456)
        self.assertEqual(event.ticker, "TEST")
        self.assertTrue(event.active)
        self.assertFalse(event.closed)

    def test_trade_model(self):
        """Test Trade model validation."""
        from agents.utils.objects import Trade
        
        trade_data = {
            "id": 1,
            "taker_order_id": "order123",
            "market": "0x123",
            "asset_id": "asset123",
            "side": "BUY",
            "size": "100",
            "fee_rate_bps": "10",
            "price": "0.5",
            "status": "MATCHED",
            "match_time": "2025-01-01T00:00:00Z",
            "last_update": "2025-01-01T00:00:00Z",
            "outcome": "Yes",
            "maker_address": "0xmaker",
            "owner": "0xowner",
            "transaction_hash": "0xtxhash",
            "bucket_index": "0",
            "maker_orders": ["order1", "order2"],
            "type": "MARKET",
        }
        trade = Trade(**trade_data)
        
        self.assertEqual(trade.id, 1)
        self.assertEqual(trade.side, "BUY")

    def test_article_model(self):
        """Test Article model for news connector."""
        from agents.utils.objects import Article, Source
        
        article_data = {
            "source": {"id": "test-source", "name": "Test Source"},
            "author": "Test Author",
            "title": "Test Article",
            "description": "Test description",
            "url": "https://example.com",
            "urlToImage": "https://example.com/image.png",
            "publishedAt": "2025-01-01T00:00:00Z",
            "content": "Test content",
        }
        article = Article(**article_data)
        
        self.assertEqual(article.title, "Test Article")
        self.assertEqual(article.source.name, "Test Source")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_parse_camel_case(self):
        """Test camelCase to space-separated conversion."""
        from agents.utils.utils import parse_camel_case
        
        self.assertEqual(parse_camel_case("testCase"), "test case")
        self.assertEqual(parse_camel_case("anotherTestCase"), "another test case")
        self.assertEqual(parse_camel_case("simple"), "simple")

    def test_preprocess_market_object(self):
        """Test market object preprocessing."""
        from agents.utils.utils import preprocess_market_object
        
        market = {
            "description": "Base description",
            "volume": 1000000,
            "liquidity": 50000,
            "active": True,
        }
        processed = preprocess_market_object(market)
        
        self.assertIn("volume", processed["description"])
        self.assertIn("liquidity", processed["description"])


class TestGammaMarketClient(unittest.TestCase):
    """Test GammaMarketClient API interactions."""

    def test_client_initialization(self):
        """Test GammaMarketClient can be instantiated."""
        from agents.polymarket.gamma import GammaMarketClient
        
        client = GammaMarketClient()
        
        self.assertEqual(client.gamma_url, "https://gamma-api.polymarket.com")
        self.assertIsNotNone(client.gamma_markets_endpoint)
        self.assertIsNotNone(client.gamma_events_endpoint)

    @patch('httpx.get')
    def test_get_markets_success(self, mock_get):
        """Test successful market retrieval."""
        from agents.polymarket.gamma import GammaMarketClient
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "id": "1",
                "question": "Test?",
                "outcomePrices": '["0.5", "0.5"]',
                "clobTokenIds": '["t1", "t2"]',
            }
        ]
        mock_get.return_value = mock_response
        
        client = GammaMarketClient()
        markets = client.get_markets()
        
        self.assertIsInstance(markets, list)

    @patch('httpx.get')
    def test_get_events_success(self, mock_get):
        """Test successful event retrieval."""
        from agents.polymarket.gamma import GammaMarketClient
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [{"id": "1", "title": "Test Event"}]
        mock_get.return_value = mock_response
        
        client = GammaMarketClient()
        events = client.get_events()
        
        self.assertIsInstance(events, list)


class TestNewsConnector(unittest.TestCase):
    """Test News connector for article retrieval."""

    def test_news_client_initialization(self):
        """Test News client can be instantiated."""
        from agents.connectors.news import News
        
        # Mock the NewsApiClient to avoid needing real API key
        with patch('agents.connectors.news.NewsApiClient'):
            client = News()
            
            self.assertIn("language", client.configs)
            self.assertIn("country", client.configs)
            self.assertIn("business", client.categories)
            self.assertIn("technology", client.categories)

    def test_get_category_mapping(self):
        """Test category mapping function."""
        from agents.connectors.news import News
        
        with patch('agents.connectors.news.NewsApiClient'):
            client = News()
            
            # Test matching category
            market_with_tech = {"category": "technology"}
            self.assertEqual(client.get_category(market_with_tech), "technology")
            
            # Test non-matching category defaults to general
            market_unknown = {"category": "unknown"}
            self.assertEqual(client.get_category(market_unknown), "general")


class TestPrompter(unittest.TestCase):
    """Test Prompter class for LLM prompt generation."""

    def test_market_analyst_prompt(self):
        """Test market analyst prompt generation."""
        from agents.application.prompts import Prompter
        
        prompter = Prompter()
        prompt = prompter.market_analyst()
        
        self.assertIn("market analyst", prompt.lower())
        self.assertIn("probability", prompt.lower())

    def test_superforecaster_prompt(self):
        """Test superforecaster prompt generation."""
        from agents.application.prompts import Prompter
        
        prompter = Prompter()
        prompt = prompter.superforecaster(
            question="Will X happen?",
            description="Test description",
            outcome="Yes"
        )
        
        self.assertIn("Will X happen?", prompt)
        self.assertIn("Test description", prompt)
        self.assertIn("Yes", prompt)
        self.assertIn("Superforecaster", prompt)

    def test_filter_events_prompt(self):
        """Test filter events prompt generation."""
        from agents.application.prompts import Prompter
        
        prompter = Prompter()
        prompt = prompter.filter_events()
        
        self.assertIn("Filter", prompt)
        self.assertIn("events", prompt.lower())

    def test_filter_markets_prompt(self):
        """Test filter markets prompt generation."""
        from agents.application.prompts import Prompter
        
        prompter = Prompter()
        prompt = prompter.filter_markets()
        
        self.assertIn("Filter", prompt)
        self.assertIn("markets", prompt.lower())

    def test_one_best_trade_prompt(self):
        """Test one best trade prompt generation."""
        from agents.application.prompts import Prompter
        
        prompter = Prompter()
        prompt = prompter.one_best_trade(
            prediction="Test prediction",
            outcomes=["Yes", "No"],
            outcome_prices=["0.5", "0.5"]
        )
        
        self.assertIn("Test prediction", prompt)
        self.assertIn("price", prompt.lower())
        self.assertIn("size", prompt.lower())

    def test_create_new_market_prompt(self):
        """Test create new market prompt."""
        from agents.application.prompts import Prompter
        
        prompter = Prompter()
        prompt = prompter.create_new_market("filtered markets data")
        
        self.assertIn("filtered markets data", prompt)
        self.assertIn("Question", prompt)
        self.assertIn("Outcomes", prompt)


class TestExecutor(unittest.TestCase):
    """Test Executor class for LLM-powered decision making."""

    @patch('agents.application.executor.ChatOpenAI')
    @patch('agents.application.executor.Polymarket')
    @patch('agents.application.executor.Chroma')
    @patch('agents.application.executor.Gamma')
    def test_executor_initialization(self, mock_gamma, mock_chroma, mock_poly, mock_llm):
        """Test Executor can be instantiated."""
        from agents.application.executor import Executor
        
        executor = Executor()
        
        self.assertIsNotNone(executor.prompter)
        self.assertIsNotNone(executor.llm)

    def test_retain_keys_function(self):
        """Test retain_keys utility function."""
        from agents.application.executor import retain_keys
        
        data = {
            "keep1": "value1",
            "keep2": "value2",
            "discard": "value3",
        }
        result = retain_keys(data, ["keep1", "keep2"])
        
        self.assertIn("keep1", result)
        self.assertIn("keep2", result)
        self.assertNotIn("discard", result)

    def test_retain_keys_nested(self):
        """Test retain_keys with nested data."""
        from agents.application.executor import retain_keys
        
        data = {
            "keep": {"nested_keep": 1, "nested_discard": 2},
            "discard": "value",
        }
        result = retain_keys(data, ["keep", "nested_keep"])
        
        self.assertIn("keep", result)
        self.assertIn("nested_keep", result["keep"])


class TestTrader(unittest.TestCase):
    """Test Trader class for trade execution."""

    @patch('agents.application.trade.Polymarket')
    @patch('agents.application.trade.Gamma')
    @patch('agents.application.trade.Agent')
    def test_trader_initialization(self, mock_agent, mock_gamma, mock_poly):
        """Test Trader can be instantiated."""
        from agents.application.trade import Trader
        
        trader = Trader()
        
        self.assertIsNotNone(trader.polymarket)
        self.assertIsNotNone(trader.gamma)
        self.assertIsNotNone(trader.agent)


class TestCreator(unittest.TestCase):
    """Test Creator class for market creation ideas."""

    @patch('agents.application.creator.Polymarket')
    @patch('agents.application.creator.Gamma')
    @patch('agents.application.creator.Agent')
    def test_creator_initialization(self, mock_agent, mock_gamma, mock_poly):
        """Test Creator can be instantiated."""
        from agents.application.creator import Creator
        
        creator = Creator()
        
        self.assertIsNotNone(creator.polymarket)
        self.assertIsNotNone(creator.gamma)
        self.assertIsNotNone(creator.agent)


class TestCLICommands(unittest.TestCase):
    """Test CLI command definitions exist."""

    def test_cli_app_exists(self):
        """Test CLI app is properly defined."""
        from scripts.python.cli import app
        import typer
        
        self.assertIsInstance(app, typer.Typer)

    def test_cli_commands_registered(self):
        """Test expected CLI commands are registered."""
        from scripts.python import cli
        
        # Verify command functions exist
        self.assertTrue(callable(cli.get_all_markets))
        self.assertTrue(callable(cli.get_relevant_news))
        self.assertTrue(callable(cli.get_all_events))
        self.assertTrue(callable(cli.create_local_markets_rag))
        self.assertTrue(callable(cli.query_local_markets_rag))
        self.assertTrue(callable(cli.ask_superforecaster))
        self.assertTrue(callable(cli.create_market))
        self.assertTrue(callable(cli.ask_llm))
        self.assertTrue(callable(cli.ask_polymarket_llm))
        self.assertTrue(callable(cli.run_autonomous_trader))


class TestServerEndpoints(unittest.TestCase):
    """Test FastAPI server endpoints."""

    def test_server_app_exists(self):
        """Test FastAPI app is properly defined."""
        from scripts.python.server import app
        from fastapi import FastAPI
        
        self.assertIsInstance(app, FastAPI)

    def test_server_routes_defined(self):
        """Test expected routes are defined."""
        from scripts.python.server import app
        
        routes = [route.path for route in app.routes]
        
        self.assertIn("/", routes)
        self.assertIn("/items/{item_id}", routes)
        self.assertIn("/trades/{trade_id}", routes)
        self.assertIn("/markets/{market_id}", routes)


class TestChromaRAG(unittest.TestCase):
    """Test Chroma RAG connector."""

    def test_chroma_rag_initialization(self):
        """Test PolymarketRAG can be instantiated."""
        from agents.connectors.chroma import PolymarketRAG
        
        rag = PolymarketRAG()
        
        self.assertIsNotNone(rag.gamma_client)


class TestIntegration(unittest.TestCase):
    """Integration tests for end-to-end workflows."""

    def test_import_all_modules(self):
        """Test all modules can be imported without errors."""
        modules_to_import = [
            'agents.application.cron',
            'agents.application.creator',
            'agents.application.trade',
            'agents.application.executor',
            'agents.application.prompts',
            'agents.connectors.news',
            'agents.connectors.search',
            'agents.connectors.chroma',
            'agents.polymarket.gamma',
            'agents.polymarket.polymarket',
            'agents.utils.objects',
            'agents.utils.utils',
            'scripts.python.cli',
            'scripts.python.server',
        ]
        
        for module_name in modules_to_import:
            with self.subTest(module=module_name):
                try:
                    __import__(module_name)
                except ImportError as e:
                    self.fail(f"Failed to import {module_name}: {e}")


if __name__ == "__main__":
    unittest.main(verbosity=2)


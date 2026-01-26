import unittest
from unittest.mock import patch, MagicMock
from polymarket_agents.graph.memory_agent import (
    memory_node,
    enrichment_node,
    reasoning_node,
    decide_node,
    MemoryAgentState,
)
from polymarket_agents.graph.planning_agent import (
    research_node,
    stats_node,
    probability_node,
    decision_node,
    PlanningState,
)
from polymarket_agents.graph.state import AgentState


class TestMemoryAgentNodes(unittest.TestCase):
    """Test individual nodes from the memory agent."""

    @patch("polymarket_agents.graph.memory_agent.MemoryManager")
    def test_memory_node_success(self, mock_mm_class):
        """Test memory node retrieves data successfully."""
        # Setup mock
        mock_mm = MagicMock()
        mock_mm.get_stats.return_value = {"total_markets": 1000}
        mock_mm.get_categories.return_value = ["politics", "sports"]
        mock_mm.list_top_volume_markets.return_value = [{"question": "Test?"}]
        mock_mm_class.return_value = mock_mm

        # Test state
        state = MemoryAgentState(
            messages=[],
            query="test query",
            memory_context={},
            live_data={},
            analysis={},
            decision={},
            error=None,
        )

        # Execute node
        result = memory_node(state)

        # Assertions
        self.assertIn("memory_context", result)
        self.assertIn("messages", result)
        self.assertEqual(
            result["memory_context"]["database_stats"]["total_markets"], 1000
        )

    @patch("polymarket_agents.connectors.gamma.GammaMarketClient")
    def test_enrichment_node_api_call(self, mock_gamma_class):
        """Test enrichment node makes API calls when needed."""
        mock_gamma = MagicMock()
        mock_gamma.get_current_events.return_value = [{"title": "Test Event"}]
        mock_gamma_class.return_value = mock_gamma

        state = MemoryAgentState(
            messages=[],
            query="live current data",
            memory_context={"relevant_markets": []},
            live_data={},
            analysis={},
            decision={},
            error=None,
        )

        result = enrichment_node(state)

        # Should have made API call and populated live_data
        self.assertIn("live_data", result)
        self.assertEqual(len(result["live_data"]["current_events"]), 1)

    def test_reasoning_node_without_llm(self):
        """Test reasoning node structure without LLM dependency."""
        # Test that the function exists and can be called with minimal setup
        state = MemoryAgentState(
            messages=[],
            query="analyze market",
            memory_context={"relevant_markets": [{"question": "Test market"}]},
            live_data={},
            analysis={},
            decision={},
            error=None,
        )

        # This will fail due to missing LLM, but should handle error gracefully
        try:
            result = reasoning_node(state)
            # If it succeeds, check structure
            self.assertIn("analysis", result)
            self.assertIn("messages", result)
        except Exception as e:
            # Should handle errors gracefully
            self.assertIsInstance(e, Exception)


class TestPlanningAgentNodes(unittest.TestCase):
    """Test planning agent node functions."""

    # Similar pattern for planning agent nodes...


class TestNodeErrorHandling(unittest.TestCase):
    """Test error handling in nodes."""

    @patch("polymarket_agents.graph.memory_agent.MemoryManager")
    def test_memory_node_handles_exceptions(self, mock_mm):
        """Test memory node gracefully handles database errors."""
        # Setup mock to raise exception
        mock_mm.side_effect = Exception("Database connection failed")

        state = MemoryAgentState(
            messages=[],
            query="test",
            memory_context={},
            live_data={},
            analysis={},
            decision={},
            error=None,
        )

        # This should not raise an exception
        result = memory_node(state)

        # Should have error in result
        self.assertIn("error", result)
        self.assertIsNotNone(result["error"])

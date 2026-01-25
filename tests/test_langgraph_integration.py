import unittest
from unittest.mock import patch, MagicMock
from polymarket_agents.graph.memory_agent import create_memory_agent
from polymarket_agents.graph.planning_agent import create_planning_agent
from langchain_core.messages import HumanMessage


class TestGraphIntegration(unittest.TestCase):
    """Test complete graph execution flows."""

    @patch("polymarket_agents.memory.manager.MemoryManager")
    @patch("polymarket_agents.connectors.gamma.GammaMarketClient")
    @patch("langchain_openai.ChatOpenAI")
    def test_memory_agent_full_flow(
        self, mock_llm_class, mock_gamma_class, mock_mm_class
    ):
        """Test complete memory agent workflow."""
        # Setup mocks
        mock_mm = MagicMock()
        mock_mm.get_stats.return_value = {"total_markets": 1000}
        mock_mm.get_categories.return_value = ["politics"]
        mock_mm.list_top_volume_markets.return_value = [{"question": "Test?"}]
        mock_mm_class.return_value = mock_mm

        mock_gamma = MagicMock()
        mock_gamma.get_current_events.return_value = [{"title": "Event"}]
        mock_gamma_class.return_value = mock_gamma

        from langchain_core.messages import AIMessage

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Analysis complete")
        mock_llm_class.return_value = mock_llm

        # Create and execute graph
        graph = create_memory_agent()

        initial_state = {
            "messages": [HumanMessage(content="analyze politics")],
            "query": "analyze politics",
            "memory_context": {},
            "live_data": {},
            "analysis": {},
            "decision": {},
            "error": None,
        }

        result = graph.invoke(initial_state)

        # Verify flow completion
        self.assertIn("analysis", result)
        self.assertIn("decision", result)
        self.assertIn("memory_context", result)

        # Verify messages were added
        self.assertGreater(len(result["messages"]), 1)

    def test_graph_state_persistence(self):
        """Test graph maintains state correctly through execution."""
        graph = create_memory_agent()

        # Run graph
        result = graph.invoke(
            {
                "messages": [HumanMessage(content="test")],
                "query": "test",
                "memory_context": {},
                "live_data": {},
                "analysis": {},
                "decision": {},
                "error": None,
            }
        )

        # Verify state structure maintained
        required_keys = [
            "messages",
            "query",
            "memory_context",
            "live_data",
            "analysis",
            "decision",
        ]
        for key in required_keys:
            self.assertIn(key, result)


class TestGraphStreaming(unittest.TestCase):
    """Test graph streaming capabilities."""

    def test_memory_agent_streaming(self):
        """Test streaming intermediate results."""
        graph = create_memory_agent()

        initial_state = {
            "messages": [HumanMessage(content="test streaming")],
            "query": "test streaming",
            "memory_context": {},
            "live_data": {},
            "analysis": {},
            "decision": {},
            "error": None,
        }

        # Test streaming mode
        stream_result = list(graph.stream(initial_state, stream_mode="updates"))

        # Should get updates for each node
        node_names = [update.keys() for update in stream_result]
        self.assertTrue(any("memory" in str(nodes) for nodes in node_names))

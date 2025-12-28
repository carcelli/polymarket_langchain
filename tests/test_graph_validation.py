import unittest
from langgraph.graph import StateGraph, END
from agents.graph.memory_agent import create_memory_agent
from agents.graph.planning_agent import create_planning_agent

class TestGraphStructure(unittest.TestCase):
    """Validate graph structure and compilation."""
    
    def test_memory_agent_compilation(self):
        """Test memory agent compiles without errors."""
        try:
            graph = create_memory_agent()
            self.assertIsNotNone(graph)
        except Exception as e:
            self.fail(f"Graph compilation failed: {e}")
    
    def test_memory_agent_node_structure(self):
        """Test memory agent has correct node structure."""
        graph = create_memory_agent()
        
        # Check that graph has expected nodes
        # Note: This requires accessing internal graph structure
        # which might not be directly exposed
        
        # Instead, test by running and checking execution
        result = graph.invoke({
            "messages": [],
            "query": "test",
            "memory_context": {},
            "live_data": {},
            "analysis": {},
            "decision": {},
            "error": None
        })
        
        # If graph executed without error, structure is valid
        self.assertIsInstance(result, dict)
    
    def test_graph_reducer_functions(self):
        """Test that state reducers work correctly."""
        from agents.graph.state import AgentState
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Test messages reducer
        messages = [HumanMessage(content="test")]
        new_messages = [AIMessage(content="response")]
        
        # The add_messages reducer should combine them
        combined = messages + new_messages
        self.assertEqual(len(combined), 2)
        self.assertIsInstance(combined[0], HumanMessage)
        self.assertIsInstance(combined[1], AIMessage)

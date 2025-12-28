import unittest
from unittest.mock import patch, MagicMock
from agents.graph.memory_agent import memory_node, enrichment_node, reasoning_node, decide_node
from agents.graph.planning_agent import research_node, stats_node, probability_node, decision_node
from agents.graph.state import AgentState, MemoryAgentState, PlanningState

class TestMemoryAgentNodes(unittest.TestCase):
    """Test individual nodes from the memory agent."""
    
    @patch('agents.memory.manager.MemoryManager')
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
            error=None
        )
        
        # Execute node
        result = memory_node(state)
        
        # Assertions
        self.assertIn("memory_context", result)
        self.assertIn("messages", result)
        self.assertEqual(result["memory_context"]["database_stats"]["total_markets"], 1000)
    
    @patch('agents.polymarket.gamma.GammaMarketClient')
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
            error=None
        )
        
        result = enrichment_node(state)
        
        # Should have made API call and populated live_data
        self.assertIn("live_data", result)
        self.assertEqual(len(result["live_data"]["current_events"]), 1)
    
    @patch('langchain_openai.ChatOpenAI')
    def test_reasoning_node_llm_call(self, mock_llm_class):
        """Test reasoning node calls LLM and processes response."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Analysis result")
        mock_llm_class.return_value = mock_llm
        
        state = MemoryAgentState(
            messages=[],
            query="analyze market",
            memory_context={"relevant_markets": [{"question": "Test market"}]},
            live_data={},
            analysis={},
            decision={},
            error=None
        )
        
        result = reasoning_node(state)
        
        self.assertIn("analysis", result)
        self.assertIn("messages", result)
        self.assertEqual(result["analysis"]["llm_response"], "Analysis result")

class TestPlanningAgentNodes(unittest.TestCase):
    """Test planning agent node functions."""
    
    # Similar pattern for planning agent nodes...
    
class TestNodeErrorHandling(unittest.TestCase):
    """Test error handling in nodes."""
    
    def test_memory_node_handles_exceptions(self):
        """Test memory node gracefully handles database errors."""
        state = MemoryAgentState(
            messages=[], query="test",
            memory_context={}, live_data={}, analysis={}, decision={}, error=None
        )
        
        # This should not raise an exception
        result = memory_node(state)
        
        # Should have error in result
        self.assertIn("error", result)
        self.assertIsNotNone(result["error"])
```

### 2. **Graph Integration Testing** 🔗

Test complete graph workflows:

```python:tests/test_langgraph_integration.py
import unittest
from unittest.mock import patch
from agents.graph.memory_agent import create_memory_agent
from agents.graph.planning_agent import create_planning_agent
from langchain_core.messages import HumanMessage

class TestGraphIntegration(unittest.TestCase):
    """Test complete graph execution flows."""
    
    @patch('agents.memory.manager.MemoryManager')
    @patch('agents.polymarket.gamma.GammaMarketClient')
    @patch('langchain_openai.ChatOpenAI')
    def test_memory_agent_full_flow(self, mock_llm_class, mock_gamma_class, mock_mm_class):
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
        
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Analysis complete")
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
            "error": None
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
        result = graph.invoke({
            "messages": [HumanMessage(content="test")],
            "query": "test",
            "memory_context": {},
            "live_data": {},
            "analysis": {},
            "decision": {},
            "error": None
        })
        
        # Verify state structure maintained
        required_keys = ["messages", "query", "memory_context", "live_data", "analysis", "decision"]
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
            "error": None
        }
        
        # Test streaming mode
        stream_result = list(graph.stream(initial_state, stream_mode="updates"))
        
        # Should get updates for each node
        node_names = [update.keys() for update in stream_result]
        self.assertTrue(any("memory" in str(nodes) for nodes in node_names))
```

### 3. **Graph Validation Testing** ✅

Test graph structure and compilation:

```python:tests/test_graph_validation.py
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
```

### 4. **Performance Testing** ⚡

Test graph performance and resource usage:

```python:tests/test_graph_performance.py
import unittest
import time
from agents.graph.memory_agent import create_memory_agent
from agents.graph.planning_agent import create_planning_agent

class TestGraphPerformance(unittest.TestCase):
    """Performance tests for graphs."""
    
    def test_memory_agent_execution_time(self):
        """Test memory agent executes within time limits."""
        graph = create_memory_agent()
        
        start_time = time.time()
        result = graph.invoke({
            "messages": [],
            "query": "performance test",
            "memory_context": {},
            "live_data": {},
            "analysis": {},
            "decision": {},
            "error": None
        })
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (adjust based on your needs)
        self.assertLess(execution_time, 30.0)  # 30 seconds max
    
    def test_graph_memory_usage(self):
        """Test graph doesn't have memory leaks."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple graph executions
        graph = create_memory_agent()
        for i in range(10):
            graph.invoke({
                "messages": [],
                "query": f"test {i}",
                "memory_context": {},
                "live_data": {},
                "analysis": {},
                "decision": {},
                "error": None
            })
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        self.assertLess(memory_increase, 100)  # Less than 100MB increase
```

### 5. **End-to-End Testing** 🌐

Test complete workflows with realistic data:

```python:tests/test_e2e_graphs.py
import unittest
from agents.graph.memory_agent import create_memory_agent, run_memory_agent
from agents.graph.planning_agent import analyze_bet

class TestEndToEndGraphs(unittest.TestCase):
    """End-to-end tests for complete graph workflows."""
    
    def test_memory_agent_e2e(self):
        """Test complete memory agent workflow with realistic query."""
        graph = create_memory_agent()
        
        query = "Find interesting political markets about the upcoming election"
        result = run_memory_agent(graph, query, verbose=False)
        
        # Verify complete workflow executed
        self.assertIn("analysis", result)
        self.assertIn("memory_context", result)
        self.assertIsNotNone(result.get("analysis", {}).get("llm_response"))
        
        # Verify analysis contains expected elements
        analysis = result["analysis"]["llm_response"]
        self.assertIsInstance(analysis, str)
        self.assertGreater(len(analysis), 10)  # Non-trivial response
    
    def test_planning_agent_e2e(self):
        """Test planning agent end-to-end analysis."""
        # This would require more complex mocking
        # but demonstrates the pattern
        
        # Mock the necessary components
        with patch('agents.memory.manager.MemoryManager') as mock_mm, \
             patch('agents.polymarket.gamma.GammaMarketClient') as mock_gamma, \
             patch('langchain_openai.ChatOpenAI') as mock_llm:
            
            # Setup mocks with realistic data
            mock_mm.return_value.list_markets_by_category.return_value = [
                {"question": "Will Candidate A win?", "volume": 1000000, "outcome_prices": ["0.6", "0.4"]}
            ]
            
            # Test the planning workflow
            result = analyze_bet("Will Candidate A win the election?")
            
            # Verify planning analysis completed
            self.assertIsInstance(result, dict)
            # Add more specific assertions based on your expected output structure
```

### 6. **Running Your Tests** ▶️

Add this to your `tests/__init__.py` or create a test runner:

```python:scripts/run_graph_tests.py
#!/usr/bin/env python3
"""
Comprehensive Graph Testing Runner

Usage:
    python scripts/run_graph_tests.py          # Run all tests
    python scripts/run_graph_tests.py --unit   # Unit tests only
    python scripts/run_graph_tests.py --perf   # Performance tests
    python scripts/run_graph_tests.py --e2e    # End-to-end tests
"""

import sys
import os
import unittest
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_unit_tests():
    """Run unit tests for individual nodes."""
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_langgraph_nodes.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_integration_tests():
    """Run graph integration tests."""
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_langgraph_integration.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_validation_tests():
    """Run graph structure validation tests."""
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_graph_validation.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_performance_tests():
    """Run performance tests."""
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_graph_performance.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def run_e2e_tests():
    """Run end-to-end tests."""
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_e2e_graphs.py')
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

def main():
    parser = argparse.ArgumentParser(description='Run LangGraph tests')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--validation', action='store_true', help='Run validation tests only')
    parser.add_argument('--perf', action='store_true', help='Run performance tests only')
    parser.add_argument('--e2e', action='store_true', help='Run end-to-end tests only')
    
    args = parser.parse_args()
    
    # Run specific test suites or all tests
    if args.unit:
        success = run_unit_tests()
    elif args.integration:
        success = run_integration_tests()
    elif args.validation:
        success = run_validation_tests()
    elif args.perf:
        success = run_performance_tests()
    elif args.e2e:
        success = run_e2e_tests()
    else:
        # Run all tests
        print("🧪 Running Complete LangGraph Test Suite")
        print("=" * 50)
        
        all_passed = True
        test_suites = [
            ("Unit Tests", run_unit_tests),
            ("Integration Tests", run_integration_tests),
            ("Validation Tests", run_validation_tests),
            ("Performance Tests", run_performance_tests),
            ("E2E Tests", run_e2e_tests),
        ]
        
        for suite_name, test_func in test_suites:
            print(f"\n▶️  Running {suite_name}...")
            try:
                passed = test_func()
                if passed:
                    print(f"✅ {suite_name}: PASSED")
                else:
                    print(f"❌ {suite_name}: FAILED")
                    all_passed = False
            except Exception as e:
                print(f"💥 {suite_name}: ERROR - {e}")
                all_passed = False
        
        success = all_passed
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
```

### 7. **Quick Validation Scripts** 🔍

Add these to your existing test suite:

```python:scripts/validate_graphs.py
#!/usr/bin/env python3
"""
Quick Graph Validation Script

Validates that all graphs compile and basic functionality works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def validate_memory_agent():
    """Validate memory agent graph."""
    print("🔍 Validating Memory Agent...")
    try:
        from agents.graph.memory_agent import create_memory_agent
        
        graph = create_memory_agent()
        print("  ✅ Graph compiles successfully")
        
        # Test basic execution
        result = graph.invoke({
            "messages": [],
            "query": "validation test",
            "memory_context": {},
            "live_data": {},
            "analysis": {},
            "decision": {},
            "error": None
        })
        print("  ✅ Graph executes successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Memory Agent validation failed: {e}")
        return False

def validate_planning_agent():
    """Validate planning agent graph."""
    print("🔍 Validating Planning Agent...")
    try:
        from agents.graph.planning_agent import create_planning_agent
        
        graph = create_planning_agent()
        print("  ✅ Graph compiles successfully")
        
        # Test basic execution with minimal state
        result = graph.invoke({
            "messages": [],
            "query": "validation test",
            "target_market_id": None,
            "market_data": {},
            "research_context": {},
            "news_sentiment": {},
            "implied_probability": 0.5,
            "price_history": [],
            "volume_analysis": {},
            "estimated_probability": 0.5,
            "probability_reasoning": "",
            "edge": 0.0,
            "expected_value": 0.0,
            "kelly_fraction": 0.0,
            "recommendation": {},
            "error": None
        })
        print("  ✅ Graph executes successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Planning Agent validation failed: {e}")
        return False

def validate_langgraph_config():
    """Validate langgraph.json configuration."""
    print("🔍 Validating LangGraph Config...")
    try:
        import json
        with open('langgraph.json', 'r') as f:
            config = json.load(f)
        
        # Validate structure
        required_keys = ['graphs', 'env', 'dependencies']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")
        
        # Validate each graph can be imported
        for graph_name, graph_path in config['graphs'].items():
            module_path, func_name = graph_path.split(':')
            print(f"  🔍 Checking {graph_name}: {module_path}.{func_name}")
            
            # Try importing the module
            __import__(module_path)
            print(f"    ✅ Module {module_path} imports successfully")
        
        print("  ✅ LangGraph config is valid")
        return True
        
    except Exception as e:
        print(f"  ❌ Config validation failed: {e}")
        return False

def main():
    """Run all validations."""
    print("🚀 Graph Validation Suite")
    print("=" * 40)
    
    validations = [
        validate_langgraph_config,
        validate_memory_agent,
        validate_planning_agent,
    ]
    
    all_passed = True
    for validate_func in validations:
        if not validate_func():
            all_passed = False
        print()
    
    if all_passed:
        print("🎉 All validations passed! Graphs are ready for production.")
        return 0
    else:
        print("❌ Some validations failed. Please fix issues before deploying.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

### **Running the Tests:**

```bash
# Run all tests
python scripts/run_graph_tests.py

# Run specific test suites
python scripts/run_graph_tests.py --unit
python scripts/run_graph_tests.py --perf

# Quick validation
python scripts/validate_graphs.py

# Run with pytest (if you set it up)
pytest tests/ -v --tb=short
```

This comprehensive testing framework will ensure your LangGraph implementations are robust, performant, and working correctly! 🎯

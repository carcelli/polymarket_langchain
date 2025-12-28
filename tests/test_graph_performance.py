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

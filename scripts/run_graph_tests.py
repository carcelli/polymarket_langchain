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
    print("🧪 Running Unit Tests...")

    # Test 1: Basic imports
    print("  📦 Testing imports...")
    try:
        from polymarket_agents.graph.state import AgentState
        from polymarket_agents.graph.memory_agent import (
            MemoryAgentState,
            memory_node,
            create_memory_agent,
        )

        print("    ✅ Core imports successful")
    except ImportError as e:
        print(f"    ❌ Import failed: {e}")
        return False

    # Test 2: State structure validation
    print("  🔍 Testing state structures...")
    try:
        state = MemoryAgentState(
            messages=[],
            query="test",
            memory_context={},
            live_data={},
            analysis={},
            decision={},
            error=None,
        )
        required_keys = [
            "messages",
            "query",
            "memory_context",
            "live_data",
            "analysis",
            "decision",
        ]
        for key in required_keys:
            assert key in state, f"Missing key: {key}"
        print("    ✅ State structures valid")
    except Exception as e:
        print(f"    ❌ State validation failed: {e}")
        return False

    # Test 3: Memory node with mocking
    print("  🧠 Testing memory node...")
    try:
        from unittest.mock import patch, MagicMock

        with patch("agents.graph.memory_agent.MemoryManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm.get_stats.return_value = {"total_markets": 1000}
            mock_mm.get_categories.return_value = ["politics", "sports"]
            mock_mm.list_top_volume_markets.return_value = [{"question": "Test?"}]
            mock_mm_class.return_value = mock_mm

            state = MemoryAgentState(
                messages=[],
                query="test query",
                memory_context={},
                live_data={},
                analysis={},
                decision={},
                error=None,
            )

            result = memory_node(state)

            assert "memory_context" in result
            assert "messages" in result
            assert result["memory_context"]["database_stats"]["total_markets"] == 1000
            print("    ✅ Memory node test passed")
    except Exception as e:
        print(f"    ❌ Memory node test failed: {e}")
        return False

    # Test 4: Graph compilation
    print("  🔗 Testing graph compilation...")
    try:
        graph = create_memory_agent()
        assert graph is not None
        print("    ✅ Graph compilation successful")
    except Exception as e:
        print(f"    ❌ Graph compilation failed: {e}")
        return False

    # Test 5: Basic graph execution
    print("  ▶️ Testing basic graph execution...")
    try:
        graph = create_memory_agent()
        result = graph.invoke(
            {
                "messages": [],
                "query": "test execution",
                "memory_context": {},
                "live_data": {},
                "analysis": {},
                "decision": {},
                "error": None,
            }
        )
        assert isinstance(result, dict)
        print("    ✅ Basic graph execution successful")
    except Exception as e:
        print(f"    ❌ Graph execution failed: {e}")
        return False

    return True


def run_integration_tests():
    """Run graph integration tests."""
    print("🔗 Running Integration Tests...")

    try:
        from polymarket_agents.graph.memory_agent import (
            create_memory_agent,
            run_memory_agent,
        )
        from unittest.mock import patch, MagicMock

        # Test full workflow
        graph = create_memory_agent()

        # Mock the memory manager to avoid database dependencies
        with patch("agents.graph.memory_agent.MemoryManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm.get_stats.return_value = {"total_markets": 500}
            mock_mm.get_categories.return_value = ["politics"]
            mock_mm.list_top_volume_markets.return_value = [
                {"question": "Test Market?"}
            ]
            mock_mm_class.return_value = mock_mm

            result = run_memory_agent(graph, "test politics", verbose=False)

            assert "analysis" in result
            assert "memory_context" in result
            print("    ✅ Integration test passed")

        return True
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def run_validation_tests():
    """Run graph structure validation tests."""
    print("✅ Running Validation Tests...")

    try:
        from polymarket_agents.graph.memory_agent import create_memory_agent

        # Test graph structures
        memory_graph = create_memory_agent()
        print("    ✅ Memory agent graph valid")

        # Basic execution test
        result = memory_graph.invoke(
            {
                "messages": [],
                "query": "validation",
                "memory_context": {},
                "live_data": {},
                "analysis": {},
                "decision": {},
                "error": None,
            }
        )

        # Check that state flows correctly
        assert "memory_context" in result
        assert "analysis" in result
        print("    ✅ Graph state flow valid")

        return True
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False


def run_performance_tests():
    """Run performance tests."""
    print("⚡ Running Performance Tests...")

    try:
        import time
        from polymarket_agents.graph.memory_agent import create_memory_agent
        from unittest.mock import patch, MagicMock

        graph = create_memory_agent()

        # Test execution time
        start_time = time.time()

        with patch("agents.graph.memory_agent.MemoryManager") as mock_mm_class:
            mock_mm = MagicMock()
            mock_mm.get_stats.return_value = {"total_markets": 100}
            mock_mm.get_categories.return_value = ["test"]
            mock_mm.list_top_volume_markets.return_value = []
            mock_mm_class.return_value = mock_mm

            result = graph.invoke(
                {
                    "messages": [],
                    "query": "performance test",
                    "memory_context": {},
                    "live_data": {},
                    "analysis": {},
                    "decision": {},
                    "error": None,
                }
            )

        execution_time = time.time() - start_time

        if execution_time < 10.0:  # Should be much faster
            print(f"    ✅ Performance test passed: {execution_time:.2f}s")
        else:
            print(f"    ❌ Performance test failed: {execution_time:.2f}s (too slow)")
            return False

        return True
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def run_e2e_tests():
    """Run end-to-end tests."""
    print("🌐 Running End-to-End Tests...")

    try:
        from polymarket_agents.graph.memory_agent import create_memory_agent
        from unittest.mock import patch, MagicMock

        # Full end-to-end test with minimal mocking
        graph = create_memory_agent()

        with patch("agents.graph.memory_agent.MemoryManager") as mock_mm_class, patch(
            "polymarket_agents.connectors.gamma.GammaMarketClient"
        ) as mock_gamma_class:

            # Setup mocks
            mock_mm = MagicMock()
            mock_mm.get_stats.return_value = {"total_markets": 1000}
            mock_mm.get_categories.return_value = ["politics", "sports"]
            mock_mm.list_top_volume_markets.return_value = [{"question": "Test?"}]
            mock_mm_class.return_value = mock_mm

            mock_gamma = MagicMock()
            mock_gamma.get_current_events.return_value = [{"title": "Test Event"}]
            mock_gamma_class.return_value = mock_gamma

            result = graph.invoke(
                {
                    "messages": [],
                    "query": "Find interesting political markets",
                    "memory_context": {},
                    "live_data": {},
                    "analysis": {},
                    "decision": {},
                    "error": None,
                }
            )

            # Verify end-to-end flow
            assert "analysis" in result
            assert "memory_context" in result
            assert "live_data" in result
            print("    ✅ End-to-end test passed")

        return True
    except Exception as e:
        print(f"❌ E2E test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run LangGraph tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--validation", action="store_true", help="Run validation tests only"
    )
    parser.add_argument(
        "--perf", action="store_true", help="Run performance tests only"
    )
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests only")

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

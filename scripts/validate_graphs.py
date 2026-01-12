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
        from polymarket_agents.graph.memory_agent import create_memory_agent

        graph = create_memory_agent()
        print("  ✅ Graph compiles successfully")

        # Test basic execution
        result = graph.invoke(
            {
                "messages": [],
                "query": "validation test",
                "memory_context": {},
                "live_data": {},
                "analysis": {},
                "decision": {},
                "error": None,
            }
        )
        print("  ✅ Graph executes successfully")
        return True

    except Exception as e:
        print(f"  ❌ Memory Agent validation failed: {e}")
        return False


def validate_planning_agent():
    """Validate planning agent graph."""
    print("🔍 Validating Planning Agent...")
    try:
        from polymarket_agents.graph.planning_agent import create_planning_agent

        graph = create_planning_agent()
        print("  ✅ Graph compiles successfully")

        # Test basic execution with minimal state
        result = graph.invoke(
            {
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
                "error": None,
            }
        )
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

        with open("langgraph.json", "r") as f:
            config = json.load(f)

        # Validate structure
        required_keys = ["graphs", "env", "dependencies"]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required key: {key}")

        # Validate each graph can be imported
        for graph_name, graph_path in config["graphs"].items():
            module_path, func_name = graph_path.split(":")
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

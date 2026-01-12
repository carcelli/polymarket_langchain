#!/usr/bin/env python3
"""
Polymarket Agent with Persistent Backends

Demonstrates how to use filesystem backends with LangGraph agents
for persistent memory and analysis storage.
"""

import sys
import os
from typing import Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parents[1] / "src"))

from polymarket_agents.backends import create_composite_backend, get_quickstart_backend
from polymarket_agents.graph.memory_agent import create_memory_agent
from polymarket_agents.graph.planning_agent import create_planning_agent, analyze_bet
from market_analysis_workflow import MarketAnalyzer


def create_agent_with_backend(backend_factory, model="gpt-4o-mini"):
    """Create a deep agent with custom backend."""
    # Import here to avoid circular imports
    try:
        from deepagents import create_deep_agent
    except ImportError:
        print("⚠️  deepagents not installed. Using standard LangGraph agent.")
        # Fallback to regular agent
        return create_memory_agent()

    return create_deep_agent(
        model=model,
        backend=backend_factory,
        tools=[],  # Add your custom tools here
    )


def demonstrate_memory_persistence():
    """Demonstrate persistent memory storage."""
    print("🧠 Memory Persistence Demo")
    print("=" * 40)

    # Create agent with filesystem backend for memories
    from polymarket_agents.backends import get_balanced_backend

    # Mock runtime for demo (in real usage, this comes from deepagents)
    class MockRuntime:
        pass

    runtime = MockRuntime()

    try:
        # Create backend
        backend_factory = get_balanced_backend(runtime, root_dir="./demo_memories")

        # Simulate storing memories
        print("📝 Storing agent memories...")

        # In real usage, this would be done through the agent's filesystem tools
        backend = backend_factory(runtime)
        if hasattr(backend, 'store_memory'):
            # Store different types of memories
            memories = [
                ("successful_analysis", "Bitcoin price predictions show 15% edge when volume > $50M", ["crypto", "edge"]),
                ("market_pattern", "Geopolitical markets resolve within 6 months 80% of time", ["geopolitics", "timing"]),
                ("risk_reminder", "Never allocate more than 2% portfolio to single market", ["risk", "portfolio"]),
            ]

            for mem_type, content, tags in memories:
                path = backend.store_memory(mem_type, content, tags)
                print(f"  ✅ Stored: {path}")

        print("\\n🔍 Retrieving memories...")
        if hasattr(backend, 'get_memories_by_type'):
            successful = backend.get_memories_by_type("successful_analysis", limit=5)
            print(f"  Found {len(successful)} successful analysis memories")

    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("💡 Install deepagents package for full backend functionality")


def demonstrate_analysis_storage():
    """Demonstrate storing analysis results persistently."""
    print("\\n📊 Analysis Storage Demo")
    print("=" * 40)

    # Use existing workflow to generate analysis
    analyzer = MarketAnalyzer()

    # Analyze a real market
    market = "Russia x Ukraine ceasefire in 2025?"
    print(f"🔍 Analyzing: {market}")

    analysis = analyzer.analyze_market_opportunity(market)

    if 'error' not in analysis:
        print("✅ Analysis complete")
        print(f"   Action: {analysis.get('action', 'UNKNOWN')}")
        print(f"   Edge: {analysis.get('edge', 0):.2f}%")

        # In a real implementation with backends, you would store this:
        print("\\n💾 Would store analysis result persistently:")
        print("   - Market intelligence")
        print("   - Statistical calculations")
        print("   - Decision reasoning")
        print("   - Performance tracking")
    else:
        print(f"❌ Analysis failed: {analysis['error']}")


def demonstrate_composite_routing():
    """Demonstrate how composite backend routes different paths."""
    print("\\n🔀 Composite Backend Routing Demo")
    print("=" * 40)

    routing_rules = {
        "/memories/": "StoreBackend (persistent)",
        "/analyses/": "StoreBackend (persistent)",
        "/strategies/": "StoreBackend (persistent)",
        "/workspace/": "FilesystemBackend (ephemeral)",
        "/market_data/": "FilesystemBackend (cached)",
        "/logs/": "FilesystemBackend (logs)",
    }

    print("📁 Path Routing Rules:")
    for path, backend in routing_rules.items():
        print(f"  {path:<12} → {backend}")

    print("\\n💡 Benefits:")
    print("  • Memories persist across sessions")
    print("  • Analysis results are searchable")
    print("  • Workspace files are temporary")
    print("  • Market data is cached locally")
    print("  • Logs are written to disk")


def show_backend_configuration():
    """Show how to configure backends for different use cases."""
    print("\\n⚙️ Backend Configuration Examples")
    print("=" * 40)

    configs = {
        "Development": {
            "backend": "get_quickstart_backend('./dev_data')",
            "description": "Local filesystem only, good for development",
            "persistence": "Single machine, survives restarts"
        },
        "Production": {
            "backend": "create_composite_backend(runtime)",
            "description": "Memories in store, workspace on disk",
            "persistence": "Cross-session, cloud-ready"
        },
        "Enterprise": {
            "backend": "create_enterprise_backend(runtime)",
            "description": "Policy-controlled with audit trails",
            "persistence": "Secure, compliant, multi-tenant"
        }
    }

    for name, config in configs.items():
        print(f"🏗️ {name}:")
        print(f"   Backend: {config['backend']}")
        print(f"   Use: {config['description']}")
        print(f"   Persistence: {config['persistence']}")
        print()


def main():
    """Main demonstration."""
    print("🚀 Polymarket Agent Backends Demo")
    print("=" * 50)
    print("Demonstrating persistent storage for trading agents")
    print()

    # Run demonstrations
    demonstrate_memory_persistence()
    demonstrate_analysis_storage()
    demonstrate_composite_routing()
    show_backend_configuration()

    print("\\n🎯 Next Steps:")
    print("1. Install deepagents: pip install deepagents")
    print("2. Choose backend configuration")
    print("3. Integrate with your agent workflows")
    print("4. Deploy with persistent memory")
    print()
    print("💡 Your agents will now learn and remember across sessions!")


if __name__ == "__main__":
    main()

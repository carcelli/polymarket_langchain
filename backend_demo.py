#!/usr/bin/env python3
"""
Backend Configuration Demo for Polymarket Deep Agents

This script demonstrates the advanced backend configurations available:
- Filesystem backends with security policies
- Composite routing for hybrid storage
- Virtual filesystems for structured workflows
- Enterprise security policies

Run with: python backend_demo.py
"""

import os
from agents.deep_research_agent import (
    virtual_demo_agent,
    enterprise_secure_agent,
    research_architect_agent
)

def demo_virtual_filesystem():
    """Demonstrate virtual filesystem capabilities."""
    print("🗂️  VIRTUAL FILESYSTEM DEMO")
    print("=" * 50)

    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")):
        print("❌ API keys required for full demo")
        return

    agent = virtual_demo_agent()

    print("\n📋 Exploring Virtual Filesystem Structure:")
    print("-" * 40)

    # List root directory
    result = agent.invoke({
        "messages": [{"role": "user", "content": "List all files and directories in the root"}]
    })
    print("Root directory contents:")
    print(result["messages"][-1].content[:500] + "...")

    # Read research template
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Read the research outline template at /templates/research_outline.md"}]
    })
    print("\n📝 Research Template Content:")
    print("-" * 30)
    print(result["messages"][-1].content[:800] + "...")

    # Show market data
    result = agent.invoke({
        "messages": [{"role": "user", "content": "Show me the active markets data"}]
    })
    print("\n📊 Market Data:")
    print("-" * 15)
    print(result["messages"][-1].content[:600] + "...")

    print("\n✅ Virtual filesystem provides structured templates and data")
    print("   Perfect for standardized research workflows!")


def demo_composite_routing():
    """Demonstrate composite backend routing."""
    print("\n🔀 COMPOSITE BACKEND ROUTING DEMO")
    print("=" * 50)

    print("\n🏗️  Advanced Routing Configuration:")
    print("-" * 35)
    print("• /workspace/ → FilesystemBackend (persistent workspace)")
    print("• /research/ → StoreBackend (cross-session research)")
    print("• /memories/ → StoreBackend (agent memories)")
    print("• /temp/ → StateBackend (ephemeral temp files)")
    print("• /cache/ → FilesystemBackend (caching layer)")
    print("• /market_data/ → StoreBackend (structured data)")

    print("\n💡 Benefits:")
    print("• Optimal storage for different data types")
    print("• Persistent research across sessions")
    print("• Fast temporary storage for intermediate results")
    print("• Structured data organization")

    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")):
        print("\n⚠️  Requires API keys for full demonstration")
        return

    agent = research_architect_agent()

    # Create files in different routes
    result = agent.invoke({
        "messages": [{"role": "user", "content": """
        Create organized research files:
        1. Write a research plan to /workspace/plan.md
        2. Save market analysis to /research/analysis.md
        3. Store agent notes in /memories/session_notes.md
        4. Cache intermediate data in /cache/temp_data.json
        5. Document market data in /market_data/metadata.json
        Then list all files to show the routing structure.
        """}]
    })

    print("\n📁 File Organization Result:")
    print("-" * 30)
    print(result["messages"][-1].content[:1000] + "...")


def demo_security_policies():
    """Demonstrate security policies and access controls."""
    print("\n🔒 SECURITY POLICIES DEMO")
    print("=" * 50)

    print("\n🛡️  Enterprise Security Features:")
    print("-" * 35)
    print("• Denied prefixes: /secrets/, /system/, /admin/, /config/")
    print("• Write/edit restrictions on sensitive paths")
    print("• Sandboxed filesystem access")
    print("• Policy enforcement at backend level")

    print("\n🚫 Blocked Operations:")
    print("• Writing to /secrets/ directory")
    print("• Editing /system/ files")
    print("• Accessing /admin/ configurations")

    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")):
        print("\n⚠️  Requires API keys for full demonstration")
        return

    agent = enterprise_secure_agent()

    # Test security policies
    result = agent.invoke({
        "messages": [{"role": "user", "content": """
        Test the security policies by trying to:
        1. Write a file to /secrets/api_keys.txt (should be blocked)
        2. Create a config file in /system/settings.json (should be blocked)
        3. Write a normal research file to /workspace/notes.md (should work)
        4. List the workspace directory to confirm normal files work
        """}]
    })

    print("\n🔐 Security Test Results:")
    print("-" * 25)
    print(result["messages"][-1].content[:800] + "...")


def demo_backend_comparison():
    """Compare different backend strategies."""
    print("\n⚖️  BACKEND STRATEGY COMPARISON")
    print("=" * 50)

    backends = {
        "Filesystem": {
            "description": "Real disk access with virtual sandboxing",
            "use_case": "Local development, persistent storage",
            "security": "Path validation, symlink protection",
            "performance": "Fast, direct disk access"
        },
        "Composite": {
            "description": "Multi-backend routing by path prefix",
            "use_case": "Complex applications needing different storage types",
            "security": "Per-route policies and restrictions",
            "performance": "Optimized for specific data types"
        },
        "Store": {
            "description": "LangGraph store for cross-session persistence",
            "use_case": "Production deployments, shared agent memory",
            "security": "Namespace isolation, access controls",
            "performance": "Database-backed durability"
        },
        "Virtual": {
            "description": "In-memory filesystem with pre-built templates",
            "use_case": "Demos, testing, structured workflows",
            "security": "No external access, controlled content",
            "performance": "Fastest, no I/O overhead"
        }
    }

    print("\n📊 Backend Comparison Matrix:")
    print("-" * 35)

    print("<15")
    print("-" * 75)

    for name, specs in backends.items():
        print("<15")

    print("\n🎯 Choosing the Right Backend:")
    print("-" * 30)
    print("• Development/Testing → Virtual or Filesystem")
    print("• Production Research → Composite with Store routing")
    print("• Enterprise Security → Filesystem with policies")
    print("• Cross-session Memory → Store backend")


def demo_advanced_routing():
    """Demonstrate advanced composite routing patterns."""
    print("\n🚦 ADVANCED ROUTING PATTERNS")
    print("=" * 50)

    print("\n🏗️  Composite Routing Strategies:")
    print("-" * 35)

    routing_patterns = {
        "Research Workflow": {
            "/workspace/": "Active project files",
            "/research/": "Completed analyses",
            "/memories/": "Agent learning",
            "/temp/": "Intermediate results"
        },
        "Enterprise Setup": {
            "/user/": "User-specific data",
            "/shared/": "Team resources",
            "/archive/": "Historical data",
            "/cache/": "Performance optimization"
        },
        "Market Analysis": {
            "/market_data/": "Structured market info",
            "/research/": "Analysis reports",
            "/models/": "Trained models",
            "/cache/": "API responses"
        }
    }

    for setup, routes in routing_patterns.items():
        print(f"\n🏢 {setup}:")
        for path, purpose in routes.items():
            print(f"  {path:<12} → {purpose}")

    print("\n💡 Routing Benefits:")
    print("• Longest-prefix matching (more specific routes win)")
    print("• Independent backend configurations per route")
    print("• Optimized storage for different data types")
    print("• Flexible scaling and migration strategies")


def main():
    """Run all backend demonstrations."""
    print("🔧 DeepAgents Backend Configuration Demo")
    print("Advanced filesystem backends for enterprise-grade agents")

    # Check environment
    has_keys = bool(os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY"))
    if not has_keys:
        print("\n⚠️  Note: Full demos require API keys")
        print("Set: ANTHROPIC_API_KEY and TAVILY_API_KEY")

    print("\n" + "=" * 60)

    # Run demos
    demo_virtual_filesystem()
    demo_composite_routing()
    demo_security_policies()
    demo_backend_comparison()
    demo_advanced_routing()

    print("\n" + "=" * 60)
    print("✅ BACKEND CONFIGURATION DEMO COMPLETE")
    print("=" * 60)

    print("""
🎯 BACKEND CONFIGURATION SUMMARY:

🔧 Storage Strategies:
• FilesystemBackend: Secure local file access with policies
• CompositeBackend: Multi-route hybrid storage
• StoreBackend: Cross-session durable persistence
• VirtualBackend: Template-based structured workflows

🛡️ Security Features:
• Path-based access controls
• Denied prefix enforcement
• Sandboxed operations
• Enterprise policy hooks

📊 Performance Optimizations:
• Route-specific storage optimization
• Fast virtual filesystem for testing
• Persistent caching layers
• Efficient data organization

🚀 Production Patterns:
• Development: Virtual + Filesystem
• Research: Composite with Store routing
• Enterprise: Policy-guarded backends
• Scaling: Route-based data distribution

🎛️ Configuration Options:
```python
# Virtual filesystem for demos
agent = create_polymarket_research_agent(storage_strategy="virtual")

# Enterprise secure setup
agent = create_polymarket_research_agent(storage_strategy="composite")

# Research architect with full routing
agent = research_architect_agent()
```

🔑 RESULT: Sophisticated backend configurations enabling
   enterprise-grade storage strategies and security policies!
""")


if __name__ == "__main__":
    main()

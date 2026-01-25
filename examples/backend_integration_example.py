#!/usr/bin/env python3
"""
Backend Integration Example for Polymarket Agents

Shows how to integrate persistent storage backends with your existing
LangGraph agents for memory and analysis persistence.
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parents[1] / "src"))

from market_analysis_workflow import MarketAnalyzer


class SimpleFilesystemBackend:
    """
    Simplified filesystem backend for demonstration.

    In production, you would use the full PolymarketFilesystemBackend
    from polymarket_agents.backends.filesystem
    """

    def __init__(self, root_dir: str = "./agent_data"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        for subdir in ["memories", "analyses", "strategies"]:
            (self.root_dir / subdir).mkdir(exist_ok=True)

    def store_memory(self, memory_type: str, content: str, tags: list = None) -> str:
        """Store an agent memory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{memory_type}_{timestamp}.json"
        filepath = self.root_dir / "memories" / filename

        memory_data = {
            "type": memory_type,
            "content": content,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(memory_data, f, indent=2)

        return str(filepath)

    def store_analysis(self, market: str, analysis: Dict[str, Any]) -> str:
        """Store market analysis result."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create safe filename
        safe_name = "".join(c for c in market[:30] if c.isalnum() or c in " _-").strip()
        safe_name = safe_name.replace(" ", "_")

        filename = f"analysis_{safe_name}_{timestamp}.json"
        filepath = self.root_dir / "analyses" / filename

        analysis_data = {
            "market": market,
            "analysis": analysis,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(analysis_data, f, indent=2, default=str)

        return str(filepath)

    def list_memories(self, memory_type: str = None) -> list:
        """List stored memories."""
        memories_dir = self.root_dir / "memories"
        if not memories_dir.exists():
            return []

        memories = []
        for file_path in memories_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                memories.append(data)
            except:
                continue

        if memory_type:
            memories = [m for m in memories if m.get("type") == memory_type]

        return sorted(memories, key=lambda x: x.get("timestamp", ""), reverse=True)

    def get_recent_analyses(self, limit: int = 5) -> list:
        """Get recent analysis results."""
        analyses_dir = self.root_dir / "analyses"
        if not analyses_dir.exists():
            return []

        analyses = []
        for file_path in analyses_dir.glob("*.json"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                analyses.append(data)
            except:
                continue

        return sorted(analyses, key=lambda x: x.get("timestamp", ""), reverse=True)[
            :limit
        ]


class AgentWithMemory:
    """Example of how to integrate memory with your existing agents."""

    def __init__(self, backend=None):
        self.backend = backend or SimpleFilesystemBackend()
        self.analyzer = MarketAnalyzer()

        # Load existing memories on startup
        self.memories = self.backend.list_memories()
        print(f"🧠 Loaded {len(self.memories)} existing memories")

    def analyze_with_memory(self, market_question: str) -> Dict[str, Any]:
        """Analyze a market and store the result."""
        print(f"🔍 Analyzing: {market_question}")

        # Run analysis
        analysis = self.analyzer.analyze_market_opportunity(market_question)

        if "error" not in analysis:
            # Store successful analysis
            stored_path = self.backend.store_analysis(market_question, analysis)
            print(f"💾 Analysis stored: {stored_path}")

            # Learn from this analysis
            self._learn_from_analysis(analysis)

        return analysis

    def _learn_from_analysis(self, analysis: Dict[str, Any]):
        """Learn patterns from analysis results."""
        action = analysis.get("action")

        if action == "PASS":
            # Learn when to pass
            edge = analysis.get("edge", 0)
            if edge < 0.5:  # Very small edge
                memory_content = f"Market showed very small edge ({edge:.2f}%) - likely not worth pursuing"
                self.backend.store_memory(
                    "low_edge_pattern", memory_content, ["edge", "pass"]
                )

        elif action in ["BUY", "SELL"]:
            # Learn successful patterns
            edge = analysis.get("edge", 0)
            if edge > 2.0:  # Good edge
                memory_content = f"Found good edge ({edge:.2f}%) in {analysis.get('category', 'unknown')} market"
                self.backend.store_memory(
                    "good_edge_pattern", memory_content, ["edge", "opportunity"]
                )

    def get_market_insights(self) -> Dict[str, Any]:
        """Get insights from stored analyses."""
        analyses = self.backend.get_recent_analyses(limit=20)

        insights = {
            "total_analyses": len(analyses),
            "categories_analyzed": set(),
            "action_distribution": {},
            "avg_edge_by_action": {},
            "recent_opportunities": [],
        }

        for analysis in analyses:
            analysis_data = analysis.get("analysis", {})

            # Track categories
            category = analysis_data.get("category", "unknown")
            insights["categories_analyzed"].add(category)

            # Track actions
            action = analysis_data.get("action", "unknown")
            insights["action_distribution"][action] = (
                insights["action_distribution"].get(action, 0) + 1
            )

            # Track edges
            edge = analysis_data.get("edge", 0)
            if action not in insights["avg_edge_by_action"]:
                insights["avg_edge_by_action"][action] = []
            insights["avg_edge_by_action"][action].append(edge)

            # Track recent opportunities
            if action in ["BUY", "SELL"] and edge > 1.0:
                insights["recent_opportunities"].append(
                    {
                        "market": analysis.get("market", ""),
                        "action": action,
                        "edge": edge,
                    }
                )

        # Calculate averages
        for action, edges in insights["avg_edge_by_action"].items():
            insights["avg_edge_by_action"][action] = (
                sum(edges) / len(edges) if edges else 0
            )

        return insights

    def show_learning_progress(self):
        """Display what the agent has learned."""
        print("\\n🧠 Agent Learning Progress")
        print("=" * 40)

        memories = self.backend.list_memories()

        # Group memories by type
        memory_types = {}
        for memory in memories:
            mem_type = memory.get("type", "unknown")
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1

        print("📚 Memory Types Learned:")
        for mem_type, count in memory_types.items():
            print(f"  • {mem_type}: {count} memories")

        # Show recent insights
        insights = self.get_market_insights()
        print(f"\\n📊 Analysis Insights:")
        print(f"  • Total analyses: {insights['total_analyses']}")
        print(f"  • Categories analyzed: {', '.join(insights['categories_analyzed'])}")

        print(f"\\n🎯 Action Distribution:")
        for action, count in insights["action_distribution"].items():
            avg_edge = insights["avg_edge_by_action"].get(action, 0)
            print(f"  • {action}: {count} times (avg edge: {avg_edge:.2f}%)")

        if insights["recent_opportunities"]:
            print(f"\\n💰 Recent Opportunities:")
            for opp in insights["recent_opportunities"][:3]:
                print(
                    f"  • {opp['market'][:40]}... - {opp['action']} (edge: {opp['edge']:.1f}%)"
                )


def demonstrate_backend_integration():
    """Demonstrate backend integration with real Polymarket data."""
    print("🚀 Backend Integration Demo")
    print("=" * 50)

    # Create agent with memory backend
    agent = AgentWithMemory()

    # Analyze some real markets
    markets_to_analyze = [
        "Russia x Ukraine ceasefire in 2025?",
        "Will the Tennessee Titans win Super Bowl 2026?",
        "Xi Jinping out in 2025?",
    ]

    print("\\n🔍 Analyzing markets and building memory...")
    for market in markets_to_analyze:
        analysis = agent.analyze_with_memory(market)
        action = analysis.get("action", "UNKNOWN")
        edge = analysis.get("edge", 0)
        print(f"  ✅ {market[:30]}...: {action} (edge: {edge:.2f}%)")

    # Show what the agent learned
    agent.show_learning_progress()

    print("\\n💾 All analyses and memories stored persistently!")
    print("   Next time you run this agent, it will remember these patterns.")


def show_backend_routing_concept():
    """Explain how composite backends work."""
    print("\\n🔀 Backend Routing Concept")
    print("=" * 50)

    routing_example = {
        "/memories/": "StoreBackend - Cross-session persistent memories",
        "/analyses/": "StoreBackend - Analysis results and patterns",
        "/strategies/": "StoreBackend - Trading strategies",
        "/workspace/": "FilesystemBackend - Temporary work files",
        "/market_data/": "FilesystemBackend - Cached market data",
        "/logs/": "FilesystemBackend - Operation logs",
    }

    print("📁 Virtual Filesystem Structure:")
    for path, description in routing_example.items():
        print(f"  {path:<12} → {description}")

    print("\\n💡 Benefits:")
    print("  • Memories persist across agent restarts")
    print("  • Analysis results are searchable and versioned")
    print("  • Temporary files don't clutter persistent storage")
    print("  • Easy to backup/restore different data types")


def main():
    """Main demonstration."""
    demonstrate_backend_integration()
    show_backend_routing_concept()

    print("\\n🎯 Next Steps:")
    print("1. Choose your backend strategy (filesystem, store, or composite)")
    print("2. Integrate with your agent workflows")
    print("3. Add memory/learning capabilities")
    print("4. Deploy with persistent storage")
    print("\\n🚀 Your agents will now learn and improve over time!")


if __name__ == "__main__":
    main()

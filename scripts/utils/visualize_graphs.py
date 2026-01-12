#!/usr/bin/env python3
"""
Graph Visualization Script for LangChain Agents

Provides multiple ways to visualize and inspect your LangGraphs:
1. ASCII diagrams (terminal)
2. Graphviz DOT files (for external visualization)
3. Interactive node/edge analysis
4. Mermaid diagrams (for documentation)
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

from polymarket_agents.graph.memory_agent import create_memory_agent
from polymarket_agents.graph.planning_agent import create_planning_agent


def visualize_ascii():
    """Show ASCII diagrams in terminal."""
    print("🖼️  ASCII Graph Visualizations")
    print("=" * 50)

    print("🔍 Memory Agent:")
    memory_graph = create_memory_agent()
    print(memory_graph.get_graph().draw_ascii())

    print("\n" + "=" * 50)
    print("🔍 Planning Agent:")
    planning_graph = create_planning_agent()
    print(planning_graph.get_graph().draw_ascii())


def analyze_graph_details():
    """Detailed node and edge analysis."""
    print("🔬 Detailed Graph Analysis")
    print("=" * 50)

    graphs = {
        "Memory Agent": create_memory_agent(),
        "Planning Agent": create_planning_agent()
    }

    for name, graph in graphs.items():
        print(f"\n📊 {name}:")
        graph_data = graph.get_graph()

        print(f"  📈 Nodes ({len(graph_data.nodes)}):")
        for node_id, node in graph_data.nodes.items():
            print(f"    • {node_id}: {node.name}")

        print(f"  🔗 Edges ({len(graph_data.edges)}):")
        for edge in graph_data.edges:
            print(f"    • {edge.source} → {edge.target}")


def generate_graphviz():
    """Generate Graphviz DOT files for visualization."""
    print("📁 Generating Graphviz DOT files...")

    graphs = {
        "memory_agent": create_memory_agent(),
        "planning_agent": create_planning_agent()
    }

    for name, graph in graphs.items():
        graph_data = graph.get_graph()

        # Generate DOT format
        dot_content = f'digraph {name} {{\n'
        dot_content += '  rankdir=TB;\n'
        dot_content += '  node [shape=box, style=rounded];\n\n'

        # Add nodes
        for node_id, node in graph_data.nodes.items():
            if node_id in ['__start__', '__end__']:
                dot_content += f'  {node_id} [shape=circle, label="{node.name}"];\n'
            else:
                dot_content += f'  {node_id} [label="{node.name}"];\n'

        dot_content += '\n'

        # Add edges
        for edge in graph_data.edges:
            dot_content += f'  {edge.source} -> {edge.target};\n'

        dot_content += '}\n'

        # Save to file
        filename = f"{name}_graph.dot"
        with open(filename, 'w') as f:
            f.write(dot_content)

        print(f"  ✅ Saved {filename}")

        # Try to generate PNG if graphviz is available
        try:
            import subprocess
            png_file = f"{name}_graph.png"
            subprocess.run(['dot', '-Tpng', filename, '-o', png_file], check=True)
            print(f"  🖼️  Generated {png_file}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  ℹ️  Install graphviz system package for PNG generation: sudo apt-get install graphviz")


def generate_mermaid():
    """Generate Mermaid diagrams for documentation."""
    print("📊 Generating Mermaid diagrams...")

    graphs = {
        "Memory Agent": create_memory_agent(),
        "Planning Agent": create_planning_agent()
    }

    for name, graph in graphs.items():
        graph_data = graph.get_graph()

        mermaid_content = "flowchart TD\n"

        # Add nodes
        for node_id, node in graph_data.nodes.items():
            if node_id in ['__start__', '__end__']:
                mermaid_content += f"    {node_id}([{node.name}])\n"
            else:
                mermaid_content += f"    {node_id}[{node.name}]\n"

        mermaid_content += "\n"

        # Add edges
        for edge in graph_data.edges:
            mermaid_content += f"    {edge.source} --> {edge.target}\n"

        # Save to file
        safe_name = name.lower().replace(" ", "_")
        filename = f"{safe_name}_mermaid.md"
        with open(filename, 'w') as f:
            f.write(f"# {name} Graph\n\n```mermaid\n{mermaid_content}```\n")

        print(f"  ✅ Saved {filename}")


def main():
    """Main visualization function."""
    print("🎨 LangGraph Visualization Suite")
    print("=" * 40)

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "ascii":
            visualize_ascii()
        elif command == "details":
            analyze_graph_details()
        elif command == "dot":
            generate_graphviz()
        elif command == "mermaid":
            generate_mermaid()
        elif command == "all":
            visualize_ascii()
            print("\n")
            analyze_graph_details()
            print("\n")
            generate_graphviz()
            print("\n")
            generate_mermaid()
        else:
            print("Usage: python visualize_graphs.py [ascii|details|dot|mermaid|all]")
    else:
        # Default: show everything
        visualize_ascii()
        print("\n")
        analyze_graph_details()
        print("\n")
        generate_graphviz()
        print("\n")
        generate_mermaid()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Data Manager CLI - Manage Polymarket ML Data Organization

Command-line interface for managing and organizing your Polymarket ML data
for optimal LangChain integration.
"""

import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

from data_organization_langchain import PolymarketDataOrganizer


def show_inventory(args):
    """Show comprehensive data inventory."""
    print("📊 Polymarket ML Data Inventory")
    print("=" * 40)

    organizer = PolymarketDataOrganizer()

    try:
        inventory = organizer.get_data_inventory()

        print("\\n📈 OVERVIEW:")
        print(f"   Total Size: {inventory['total_size_mb']:.1f} MB")
        print(f"   Databases: {len(inventory['databases'])}")
        print(f"   JSON Files: {len(inventory['json_files'])}")
        print(f"   Generated: {inventory['timestamp'][:19]}")

        print("\\n🗄️ DATABASES:")
        for db_name, db_info in inventory["databases"].items():
            print(f"   📁 {db_name} ({db_info['size_mb']:.1f} MB)")
            print(
                f"      Tables: {len(db_info['tables'])} ({', '.join(db_info['tables'])})"
            )
            total_records = sum(db_info["record_counts"].values())
            print(f"      Records: {total_records:,}")
            if "error" in db_info:
                print(f"      ⚠️ Error: {db_info['error']}")

        print("\\n📋 DATA BREAKDOWN:")
        breakdown = inventory["data_breakdown"]
        for data_type, info in breakdown.items():
            if info["count"] > 0:
                name = data_type.replace("_", " ").title()
                print(f"   • {name}: {info['count']:,} {info['description']}")

        if inventory["vector_stores"]:
            print("\\n🔍 VECTOR STORES:")
            for store_name, size_mb in inventory["vector_stores"].items():
                print(f"   📚 {store_name}: {size_mb:.1f} MB")

        print("\\n💾 RAW FILES:")
        for json_name, json_info in inventory["json_files"].items():
            print(f"   📄 {json_name}: {json_info['size_kb']:.1f} KB")

    except Exception as e:
        print(f"❌ Error getting inventory: {e}")


def show_langchain_guide(args):
    """Show LangChain integration guide."""
    print("🔗 LangChain Integration Guide")
    print("=" * 35)

    organizer = PolymarketDataOrganizer()
    guide = organizer.get_langchain_integration_guide()

    print(guide)


def create_vector_stores(args):
    """Create vector stores for semantic search."""
    print("🔍 Creating Vector Stores for LangChain")
    print("=" * 40)

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        print("   Please set it with: export OPENAI_API_KEY='your-key-here'")
        print("   Or get one from: https://platform.openai.com/api-keys")
        return

    organizer = PolymarketDataOrganizer()

    try:
        print("⏳ Creating vector stores... (this may take a few minutes)")

        stores_created = organizer.create_vector_stores()

        print("\\n✅ VECTOR STORES CREATED:")
        for store_name, store_info in stores_created.items():
            print(f"   📚 {store_name}: {store_info['documents']:,} documents")
            print(f"      Path: {store_info['path']}")

        print("\\n🛠️ LANGCHAIN TOOLS NOW AVAILABLE:")
        tools = organizer.create_langchain_tools()
        print(f"   • {len(tools)} tools created for agent integration")

        # Show usage example
        print("\\n💡 USAGE EXAMPLE:")
        print("   from data_organization_langchain import PolymarketDataOrganizer")
        print("   organizer = PolymarketDataOrganizer()")
        print("   tools = organizer.create_langchain_tools()")
        print("   # Use tools['search_markets'] in your LangChain agent")

    except Exception as e:
        print(f"❌ Error creating vector stores: {e}")
        if "embedding" in str(e).lower():
            print("   This might be due to OpenAI API limits or network issues")
            print("   Try again later or check your API quota")


def analyze_growth(args):
    """Analyze data growth patterns and projections."""
    print("📈 Data Growth Analysis")
    print("=" * 25)

    organizer = PolymarketDataOrganizer()
    inventory = organizer.get_data_inventory()

    print("\\n📊 CURRENT DATA SCALE:")
    print(f"   Markets: {inventory['data_breakdown']['markets_data']['count']:,}")
    print(
        f"   ML Experiments: {inventory['data_breakdown']['ml_experiments']['count']}"
    )
    print(f"   ML Models: {inventory['data_breakdown']['ml_models']['count']}")
    print(f"   Workflows: {inventory['data_breakdown']['workflows']['count']}")
    print(f"   Total Size: {inventory['total_size_mb']:.1f} MB")

    print("\\n📈 GROWTH PROJECTIONS:")
    print("   Assuming weekly ML workflows + daily market updates:")
    print("   ")
    print("   Week 1: +5 workflows, +1000 markets = ~32 MB")
    print("   Month 1: +20 workflows, +4000 markets = ~45 MB")
    print("   Quarter 1: +80 workflows, +16000 markets = ~120 MB")
    print("   Year 1: +400 workflows, +80000 markets = ~600 MB")

    print("\\n🗂️ ORGANIZATION RECOMMENDATIONS:")
    print("   • Vector stores: Rebuild monthly for optimal performance")
    print("   • Database partitioning: Consider by time period after 6 months")
    print("   • Archival: Move old experiments to cold storage after 1 year")
    print("   • Backup: Daily database backups, weekly full system backups")

    print("\\n💡 SCALING STRATEGIES:")
    print("   • Distributed vector stores (Pinecone/Weaviate) for >1GB data")
    print("   • Database sharding by category/time for >1M markets")
    print("   • Caching layer for frequent queries")
    print("   • Async processing for vector store updates")


def optimize_storage(args):
    """Optimize data storage and organization."""
    print("🔧 Data Storage Optimization")
    print("=" * 32)

    organizer = PolymarketDataOrganizer()

    print("\\n🧹 CLEANUP OPTIONS:")
    print("   • Remove old workflow reports (>30 days): Saves ~1MB/month")
    print("   • Archive completed experiments: Saves ~5MB/month")
    print("   • Compress vector stores: Saves ~20-30% space")
    print("   • Vacuum SQLite databases: Reclaims unused space")

    print("\\n📁 RECOMMENDED STRUCTURE:")
    print("   data/")
    print("   ├── markets.db              # Main market data")
    print("   ├── standalone_ml.db         # ML experiments/models")
    print("   ├── memory.db               # Agent conversations")
    print("   └── ingested_*.json         # Raw API responses")
    print("   ")
    print("   vector_stores/")
    print("   ├── markets_vectorstore/    # Market semantic search")
    print("   ├── ml_results_vectorstore/ # ML results search")
    print("   └── combined_vectorstore/   # Unified search")
    print("   ")
    print("   archives/")
    print("   └── YYYY-MM/               # Monthly archives")

    print("\\n🗜️ COMPRESSION SAVINGS:")
    print("   • SQLite VACUUM: Reclaims 10-50% unused space")
    print("   • Vector store compression: 20-30% reduction")
    print("   • JSON gzip compression: 70-80% for archives")
    print("   • Total potential: 40-60% space reduction")

    # Show actual optimization commands
    print("\\n🛠️ OPTIMIZATION COMMANDS:")
    print("   # Vacuum SQLite databases")
    print("   sqlite3 data/markets.db 'VACUUM;'")
    print("   sqlite3 data/standalone_ml.db 'VACUUM;'")
    print("   ")
    print("   # Compress old JSON files")
    print("   gzip data/ingested_*.json")
    print("   ")
    print("   # Archive old data")
    print("   mkdir -p archives/$(date +%Y-%m)")
    print("   mv workflow_report_old_*.json archives/$(date +%Y-%m)/")


def show_langchain_tools(args):
    """Show available LangChain tools."""
    print("🛠️ LangChain Tools Available")
    print("=" * 30)

    try:
        organizer = PolymarketDataOrganizer()
        tools = organizer.create_langchain_tools()

        print(f"\\n✅ {len(tools)} tools created for LangChain agents:")
        print()

        tool_categories = {
            "search": "Vector Search Tools",
            "query": "Database Query Tools",
            "get": "Utility Tools",
        }

        for category, title in tool_categories.items():
            category_tools = {k: v for k, v in tools.items() if k.startswith(category)}
            if category_tools:
                print(f"🔍 {title}:")
                for tool_name in category_tools.keys():
                    print(f"   • {tool_name}")
                print()

        print("💡 USAGE IN LANGCHAIN AGENT:")
        print("   from langchain.agents import create_react_agent")
        print("   ")
        print("   tools = organizer.create_langchain_tools()")
        print("   agent = create_react_agent(")
        print("       llm=your_llm,")
        print("       tools=list(tools.values()),")
        print("       prompt=your_prompt")
        print("   )")
        print("   ")
        print("   # Agent can now search and query all your data!")
        print("   response = agent.run('Find high-volume crypto markets')")

    except Exception as e:
        print(f"❌ Error creating tools: {e}")
        print("   (This may require OPENAI_API_KEY for vector stores)")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Polymarket ML Data Manager - Organize data for LangChain integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # See what data you have
  python data_manager_cli.py inventory

  # Create vector stores for semantic search (requires OPENAI_API_KEY)
  python data_manager_cli.py create-vectors

  # See LangChain integration options
  python data_manager_cli.py langchain-guide

  # Analyze data growth projections
  python data_manager_cli.py analyze-growth

  # Optimize storage and organization
  python data_manager_cli.py optimize

  # Show available LangChain tools
  python data_manager_cli.py tools
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Inventory command
    subparsers.add_parser("inventory", help="Show comprehensive data inventory")

    # LangChain guide
    subparsers.add_parser("langchain-guide", help="Show LangChain integration guide")

    # Create vector stores
    subparsers.add_parser(
        "create-vectors", help="Create vector stores for semantic search"
    )

    # Analyze growth
    subparsers.add_parser("analyze-growth", help="Analyze data growth patterns")

    # Optimize storage
    subparsers.add_parser("optimize", help="Show storage optimization options")

    # Show tools
    subparsers.add_parser("tools", help="Show available LangChain tools")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "inventory":
            show_inventory(args)
        elif args.command == "langchain-guide":
            show_langchain_guide(args)
        elif args.command == "create-vectors":
            create_vector_stores(args)
        elif args.command == "analyze-growth":
            analyze_growth(args)
        elif args.command == "optimize":
            optimize_storage(args)
        elif args.command == "tools":
            show_langchain_tools(args)

    except KeyboardInterrupt:
        print("\\n\\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

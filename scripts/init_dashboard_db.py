#!/usr/bin/env python3
"""
Initialize database tables for dashboard execution tracking.

This script ensures all required tables exist in data/markets.db
for tracking agent executions, performance metrics, and system health.

Usage:
    python scripts/init_dashboard_db.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from polymarket_agents.memory.manager import MemoryManager


def main():
    """Initialize database tables."""
    print("🔧 Initializing dashboard database tables...")
    print("=" * 60)

    try:
        # MemoryManager.__init__ calls _init_db which creates all tables
        memory = MemoryManager()

        # Verify tables exist
        stats = memory.get_database_stats()

        print("✅ Database initialized successfully!")
        print(f"   Database: {memory.db_path}")
        print(f"   Total markets: {stats.get('total_markets', 0):,}")
        print(f"   Database size: {stats.get('database_size_mb', 0):.2f} MB")
        print("\n💡 Next steps:")
        print("   1. Run an agent to generate tracking data:")
        print(
            "      python scripts/python/cli.py run-memory-agent 'Find crypto markets'"
        )
        print("   2. Launch the dashboard:")
        print("      python scripts/python/cli.py dashboard")

    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

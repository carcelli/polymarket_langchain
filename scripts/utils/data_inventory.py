#!/usr/bin/env python3
"""
Simple Data Inventory - Show what data you're creating

Quick overview of your Polymarket ML data without complex dependencies.
"""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime


def get_data_inventory():
    """Get comprehensive inventory of all data."""
    data_dir = Path("./data")
    inventory = {
        'timestamp': datetime.now().isoformat(),
        'total_size_mb': 0,
        'databases': {},
        'json_files': {},
        'data_breakdown': {}
    }

    # Check databases
    db_files = list(data_dir.glob("*.db"))
    for db_file in db_files:
        size_mb = db_file.stat().st_size / (1024 * 1024)

        db_info = {
            'size_mb': round(size_mb, 2),
            'tables': [],
            'record_counts': {}
        }

        try:
            with sqlite3.connect(str(db_file)) as conn:
                # Get table names
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
                db_info['tables'] = [t[0] for t in tables]

                # Get record counts
                for table in db_info['tables']:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                    db_info['record_counts'][table] = count

        except Exception as e:
            db_info['error'] = str(e)

        inventory['databases'][db_file.name] = db_info
        inventory['total_size_mb'] += size_mb

    # Check JSON files
    json_files = list(data_dir.glob("*.json")) + list(Path(".").glob("workflow_report_*.json"))
    for json_file in json_files:
        size_kb = json_file.stat().st_size / 1024

        inventory['json_files'][json_file.name] = {
            'size_kb': round(size_kb, 2),
            'path': str(json_file)
        }
        inventory['total_size_mb'] += size_kb / 1024

    # Data breakdown analysis
    breakdown = {
        'markets_data': {'count': 0, 'description': 'Polymarket trading data'},
        'ml_experiments': {'count': 0, 'description': 'ML model training experiments'},
        'ml_models': {'count': 0, 'description': 'Trained ML models'},
        'workflows': {'count': 0, 'description': 'Automated ML workflow executions'},
        'reports': {'count': 0, 'description': 'Generated analysis reports'},
        'memory': {'count': 0, 'description': 'Agent conversation memory'}
    }

    # Markets data
    markets_db = data_dir / "markets.db"
    if markets_db.exists():
        try:
            with sqlite3.connect(str(markets_db)) as conn:
                breakdown['markets_data']['count'] = conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0]
        except:
            pass

    # ML data
    ml_db = data_dir / "standalone_ml.db"
    if ml_db.exists():
        try:
            with sqlite3.connect(str(ml_db)) as conn:
                breakdown['ml_experiments']['count'] = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
                breakdown['ml_models']['count'] = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
                breakdown['workflows']['count'] = conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0]
        except:
            pass

    # Reports
    reports = list(Path(".").glob("workflow_report_*.json"))
    breakdown['reports']['count'] = len(reports)

    # Memory
    memory_db = data_dir / "memory.db"
    if memory_db.exists():
        breakdown['memory']['count'] = 1

    inventory['data_breakdown'] = breakdown

    return inventory


def show_inventory():
    """Show comprehensive data inventory."""
    print("📊 Polymarket ML Data Inventory")
    print("=" * 40)

    inventory = get_data_inventory()

    print("
📈 OVERVIEW:"    print(f"   Total Size: {inventory['total_size_mb']:.1f} MB")
    print(f"   Databases: {len(inventory['databases'])}")
    print(f"   JSON Files: {len(inventory['json_files'])}")
    print(f"   Generated: {inventory['timestamp'][:19]}")

    print("
🗄️ DATABASES:"    for db_name, db_info in inventory['databases'].items():
        print(f"   📁 {db_name} ({db_info['size_mb']:.1f} MB)")
        print(f"      Tables: {len(db_info['tables'])} ({', '.join(db_info['tables'])})")
        total_records = sum(db_info['record_counts'].values())
        print(f"      Records: {total_records:,}")
        if 'error' in db_info:
            print(f"      ⚠️ Error: {db_info['error']}")

    print("
📋 DATA BREAKDOWN:"    breakdown = inventory['data_breakdown']
    for data_type, info in breakdown.items():
        if info['count'] > 0:
            name = data_type.replace('_', ' ').title()
            print(f"   • {name}: {info['count']:,} {info['description']}")

    print("
💾 RAW FILES:"    for json_name, json_info in inventory['json_files'].items():
        print(f"   📄 {json_name}: {json_info['size_kb']:.1f} KB")

    print("
🔍 LANGCHAIN INTEGRATION:"    print("   Your data is ready for LangChain agents!")
    print("   Use semantic search over 20K+ markets + ML results")
    print("   Query structured data with SQL-like precision")
    print("   Combine market analysis with ML insights")

    print("
💡 NEXT STEPS:"    print("   1. Run workflows: python standalone_ml_workflow.py")
    print("   2. Add OpenAI key for vector search")
    print("   3. Create vector stores for semantic search")
    print("   4. Build LangChain agents with your data")


if __name__ == "__main__":
    show_inventory()

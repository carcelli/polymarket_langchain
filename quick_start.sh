# Quick Start Commands for Real Polymarket Analysis

# 1. Check available markets
python -c "
import sqlite3
conn = sqlite3.connect('data/markets.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM markets WHERE active = 1')
print(f'Active markets: {cursor.fetchone()[0]}')
conn.close()
"

# 2. Analyze specific market
python scripts/python/cli.py run-planning-agent 'Russia x Ukraine ceasefire in 2025?'

# 3. Find crypto opportunities  
python scripts/python/cli.py run-memory-agent 'Find interesting crypto markets'

# 4. Run comprehensive workflow
python market_analysis_workflow.py

# 5. Check market volume leaders
python -c "
import sqlite3
conn = sqlite3.connect('data/markets.db')
cursor = conn.cursor()
cursor.execute('SELECT question, volume FROM markets WHERE active = 1 ORDER BY volume DESC LIMIT 3')
for row in cursor.fetchall():
    print(f'{row[0][:50]}... - ${row[1]:,.0f}')
conn.close()
"


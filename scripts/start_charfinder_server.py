#!/usr/bin/env python3
"""
Character Finder Server Startup Script

Starts the asyncio TCP character finder server for Unicode name lookups.
This provides a lightweight internal service for agent metadata enrichment.

Usage:
    python scripts/start_charfinder_server.py [port] [host]

Examples:
    python scripts/start_charfinder_server.py 2323
    python scripts/start_charfinder_server.py 2323 0.0.0.0  # Bind to all interfaces

The server will:
- Build/load Unicode character index on first run (~10 seconds initially)
- Listen for TCP connections on specified host:port
- Handle concurrent client queries using asyncio coroutines
- Cache index to disk for fast subsequent startups

Test the server:
    telnet localhost 2323
    ?> chess
    ?> arrow
    ?> greek
    (Ctrl+C to exit telnet)

Agents can query via:
    from polymarket_agents.utils.charfinder_client import query_unicode_names
    results = query_unicode_names("bitcoin")  # Might find ₿ symbol
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from polymarket_agents.utils.charfinder_server import main

if __name__ == '__main__':
    # Parse arguments: [host] [port]
    if len(sys.argv) >= 3:
        host, port = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 2:
        host, port = '127.0.0.1', sys.argv[1]
    else:
        host, port = '127.0.0.1', '2323'

    print("🚀 Starting Character Finder Server")
    print(f"📍 Binding to: {host}:{port}")
    print("💡 Test with: telnet localhost 2323")
    print("🛑 Stop with: Ctrl+C")
    print("-" * 50)

    try:
        main(host, port)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)
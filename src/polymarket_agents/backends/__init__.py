"""
Filesystem Backends for Polymarket LangGraph Agents

This module provides persistent storage backends for agents to maintain
memories, analysis results, and knowledge across sessions.
"""

from .composite import create_composite_backend
from .filesystem import PolymarketFilesystemBackend
from .store import PolymarketStoreBackend

__all__ = [
    'create_composite_backend',
    'PolymarketFilesystemBackend',
    'PolymarketStoreBackend'
]

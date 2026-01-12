"""
Composite Backend for Polymarket Agents

Combines multiple backends with path-based routing for optimal data management.
"""

from typing import Dict, Any, Callable, TYPE_CHECKING
from deepagents.backends import CompositeBackend as BaseCompositeBackend

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore

from .filesystem import PolymarketFilesystemBackend
from .store import PolymarketStoreBackend


def create_composite_backend(
    runtime,
    root_dir: str = "./agent_data",
    store_namespace: str = "polymarket_agent"
) -> Callable:
    """
    Create a composite backend optimized for Polymarket agents.

    Routing strategy:
    - /memories/* → StoreBackend (persistent across sessions)
    - /analyses/* → StoreBackend (important analysis results)
    - /strategies/* → StoreBackend (trading strategies)
    - /workspace/* → FilesystemBackend (temporary work files)
    - /market_data/* → FilesystemBackend (cached data)
    - /logs/* → FilesystemBackend (operation logs)

    Args:
        runtime: ToolRuntime for accessing store
        root_dir: Root directory for filesystem backend
        store_namespace: Namespace for store backend

    Returns:
        Backend factory function for use with create_deep_agent
    """

    def backend_factory(rt):
        # Create individual backends
        filesystem_backend = PolymarketFilesystemBackend(root_dir=root_dir, virtual_mode=True)
        store_backend = PolymarketStoreBackend(rt, namespace=store_namespace)

        # Define routing rules
        routes = {
            "/memories/": store_backend,      # Long-term memories
            "/analyses/": store_backend,      # Analysis results
            "/strategies/": store_backend,    # Trading strategies
            "/patterns/": store_backend,      # Learned patterns
        }

        # Default to filesystem for everything else
        return BaseCompositeBackend(
            default=filesystem_backend,
            routes=routes
        )

    return backend_factory


def create_memory_focused_backend(runtime, memory_dir: str = "./agent_memories") -> Callable:
    """
    Create a backend focused on memory persistence.

    - /memories/* → FilesystemBackend (local persistence)
    - Everything else → StateBackend (ephemeral)
    """
    from deepagents.backends import StateBackend, FilesystemBackend

    def backend_factory(rt):
        memory_backend = PolymarketFilesystemBackend(
            root_dir=memory_dir,
            virtual_mode=True
        )

        return BaseCompositeBackend(
            default=StateBackend(rt),
            routes={
                "/memories/": memory_backend,
            }
        )

    return backend_factory


def create_enterprise_backend(
    runtime,
    root_dir: str = "./agent_data",
    store_namespace: str = "polymarket_agent",
    deny_writes: list = None
) -> Callable:
    """
    Create an enterprise-grade backend with policy controls.

    Features:
    - Comprehensive routing
    - Write protection for sensitive paths
    - Audit logging
    """
    from deepagents.backends import PolicyWrapper

    deny_writes = deny_writes or ["/system/", "/config/"]

    def backend_factory(rt):
        # Create base composite backend
        base_backend = create_composite_backend(rt, root_dir, store_namespace)(rt)

        # Add policy wrapper for security
        return PolicyWrapper(
            inner=base_backend,
            deny_prefixes=deny_writes
        )

    return backend_factory


# Convenience functions for common configurations
def get_quickstart_backend(root_dir: str = "./agent_data") -> Callable:
    """Quick filesystem-only backend for development."""
    def backend_factory(rt):
        return PolymarketFilesystemBackend(root_dir=root_dir, virtual_mode=True)
    return backend_factory


def get_persistent_backend(runtime, store_namespace: str = "polymarket_agent") -> Callable:
    """Store-only backend for cloud deployments."""
    def backend_factory(rt):
        return PolymarketStoreBackend(rt, namespace=store_namespace)
    return backend_factory


def get_balanced_backend(runtime, root_dir: str = "./agent_data") -> Callable:
    """Balanced backend: memories persistent, workspace ephemeral."""
    return create_memory_focused_backend(runtime, root_dir)

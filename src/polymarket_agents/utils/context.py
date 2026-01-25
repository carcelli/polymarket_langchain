"""
Context Engineering for Polymarket Agents

This module provides structures for managing different types of context:
1. Runtime Context: Static configuration (API keys, environment).
2. State: Short-term, conversation-scoped memory (tool results, flags).
3. Store: Long-term, cross-conversation memory (user preferences, market insights).
"""

import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from polymarket_agents.memory.manager import MemoryManager


@dataclass
class RuntimeContext:
    """Static configuration and environment settings."""

    user_id: str = "default_user"
    user_role: str = "trader"
    deployment_env: str = "development"
    api_keys: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls):
        return cls(
            api_keys={
                "openai": os.getenv("OPENAI_API_KEY"),
                "tavily": os.getenv("TAVILY_API_KEY"),
                "newsapi": os.getenv("NEWSAPI_API_KEY"),
            }
        )


class ContextManager:
    """
    Orchestrates different context sources to provide the 'right' information to agents.
    """

    def __init__(
        self,
        runtime: Optional[RuntimeContext] = None,
        memory_db: str = "data/markets.db",
    ):
        self.runtime = runtime or RuntimeContext.from_env()
        self.state: Dict[str, Any] = {
            "start_time": datetime.now().isoformat(),
            "messages_count": 0,
            "tool_results": [],
        }
        self.store = MemoryManager(memory_db)

    def update_state(self, updates: Dict[str, Any]):
        """Update short-term conversation state."""
        self.state.update(updates)

    def get_model_context(self) -> str:
        """
        Generates a context string to be injected into the system prompt.
        This follows the 'Context Engineering' principle of providing the right data format.
        """
        # 1. Access Runtime Context
        role_info = f"You are acting as a {self.runtime.user_role} in a {self.runtime.deployment_env} environment."

        # 2. Access Store (Long-term insights)
        stats = self.store.get_stats()
        store_info = f"Our local market database contains {stats['total_markets']} active markets across {stats['total_categories']} categories."

        # 3. Access State (Short-term)
        state_info = f"Current session message count: {self.state['messages_count']}."
        if self.state.get("authenticated"):
            state_info += " User is AUTHENTICATED for trading."
        else:
            state_info += " User is currently UNAUTHENTICATED."

        return f"""
### CONTEXTUAL INFORMATION
{role_info}
{store_info}
{state_info}

### INSTRUCTIONS
- If session is long (>10 messages), be extra concise.
- Always check the local database stats before making broad claims.
- In production environment, be extra cautious with execution tools.
"""

    def inject_context_into_prompt(self, prompt_template: str) -> str:
        """Helper to manually inject context if not using LangChain middleware."""
        context_block = self.get_model_context()
        if "{context}" in prompt_template:
            return prompt_template.replace("{context}", context_block)
        return f"{context_block}\n\n{prompt_template}"

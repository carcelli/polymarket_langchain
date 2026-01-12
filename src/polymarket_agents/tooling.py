"""
Tool wrapping helpers for LangChain-compatible tools.

Provides a minimal fallback wrapper when LangChain is unavailable.
"""

import inspect
from typing import Any, Callable, Dict, Optional, Type

from pydantic import BaseModel

try:
    from langchain_core.tools import StructuredTool

    _LANGCHAIN_AVAILABLE = True
except Exception:
    StructuredTool = None
    _LANGCHAIN_AVAILABLE = False


class ToolWrapper:
    """Minimal tool wrapper for environments without LangChain."""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        args_schema: Optional[Type[BaseModel]] = None,
    ) -> None:
        self.func = func
        self.name = name or func.__name__
        self.description = description or inspect.getdoc(func) or ""
        self.args_schema = args_schema

    def invoke(self, input: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        if input is None:
            return self.func(**kwargs)
        if isinstance(input, dict):
            return self.func(**input)
        return self.func(input)

    def run(self, input: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        return self.invoke(input, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        return self.func(*args, **kwargs)


def wrap_tool(
    func: Callable,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    args_schema: Optional[Type[BaseModel]] = None,
):
    tool_name = name or func.__name__
    tool_description = description or inspect.getdoc(func) or ""
    if _LANGCHAIN_AVAILABLE:
        return StructuredTool.from_function(
            func=func,
            name=tool_name,
            description=tool_description,
            args_schema=args_schema,
        )
    return ToolWrapper(
        func=func,
        name=tool_name,
        description=tool_description,
        args_schema=args_schema,
    )

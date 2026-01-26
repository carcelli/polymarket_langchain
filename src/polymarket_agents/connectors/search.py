from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class SearchConfig:
    tavily_api_key: str

    @classmethod
    def from_env(cls) -> "SearchConfig":
        key = os.getenv("TAVILY_API_KEY")
        if not key:
            raise ValueError("TAVILY_API_KEY required")
        return cls(tavily_api_key=key)


class Search:
    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig.from_env()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from tavily import TavilyClient  # type: ignore[import]
            except ImportError as e:
                raise ImportError("pip install tavily-python") from e
            self._client = TavilyClient(api_key=self.config.tavily_api_key)
        return self._client

    def search_context(self, query: str):
        return self._get_client().get_search_context(query=query)


search_client = Search()

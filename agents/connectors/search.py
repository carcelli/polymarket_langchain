import os

from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPEN_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Lazy import to avoid import errors when package not available
TAVILY_AVAILABLE = False
_tavily_client_class = None

def _get_tavily_client():
    global _tavily_client_class, TAVILY_AVAILABLE
    if _tavily_client_class is None:
        try:
            from tavily import TavilyClient
            _tavily_client_class = TavilyClient
            TAVILY_AVAILABLE = True
        except ImportError:
            _tavily_client_class = None
            TAVILY_AVAILABLE = False
            raise ImportError("tavily-python package is not installed. Install with: pip install tavily-python")
    return _tavily_client_class


class Search:
    def __init__(self):
        self._tavily_client = None

    @property
    def tavily_client(self):
        if self._tavily_client is None:
            if not tavily_api_key:
                raise ValueError("TAVILY_API_KEY environment variable is required")
            TavilyClient = _get_tavily_client()
            self._tavily_client = TavilyClient(api_key=tavily_api_key)
        return self._tavily_client

    def search_context(self, query: str):
        """Execute a context search query"""
        return self.tavily_client.get_search_context(query=query)


# Create singleton instance
search_client = Search()

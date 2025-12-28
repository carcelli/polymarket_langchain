from datetime import datetime
import os

from agents.utils.objects import Article

# Lazy import to avoid import errors when package not available
NEWSAPI_AVAILABLE = False
_newsapi_client_class = None

def _get_newsapi_client():
    global _newsapi_client_class, NEWSAPI_AVAILABLE
    if _newsapi_client_class is None:
        try:
            from newsapi import NewsApiClient
            _newsapi_client_class = NewsApiClient
            NEWSAPI_AVAILABLE = True
        except ImportError:
            _newsapi_client_class = None
            NEWSAPI_AVAILABLE = False
            raise ImportError("newsapi-python package is not installed. Install with: pip install newsapi-python")
    return _newsapi_client_class


class News:
    def __init__(self) -> None:
        self.configs = {
            "language": "en",
            "country": "us",
            "top_headlines": "https://newsapi.org/v2/top-headlines?country=us&apiKey=",
            "base_url": "https://newsapi.org/v2/",
        }

        self.categories = {
            "business",
            "entertainment",
            "general",
            "health",
            "science",
            "sports",
            "technology",
        }

        self._api = None

    @property
    def API(self):
        if self._api is None:
            api_key = os.getenv("NEWSAPI_API_KEY")
            if not api_key:
                raise ValueError("NEWSAPI_API_KEY environment variable is required")
            NewsApiClient = _get_newsapi_client()
            self._api = NewsApiClient(api_key)
        return self._api

    def get_articles_for_cli_keywords(self, keywords) -> "list[Article]":
        query_words = keywords.split(",")
        all_articles = self.get_articles_for_options(query_words)
        article_objects: list[Article] = []
        for _, articles in all_articles.items():
            for article in articles:
                article_objects.append(Article(**article))
        return article_objects

    def get_top_articles_for_market(self, market_object: dict) -> "list[Article]":
        return self.API.get_top_headlines(
            language="en", country="usa", q=market_object["description"]
        )

    def get_articles_for_options(
        self,
        market_options: "list[str]",
        date_start: datetime = None,
        date_end: datetime = None,
    ) -> "list[Article]":

        all_articles = {}
        # Default to top articles if no start and end dates are given for search
        if not date_start and not date_end:
            for option in market_options:
                response_dict = self.API.get_top_headlines(
                    q=option.strip(),
                    language=self.configs["language"],
                    country=self.configs["country"],
                )
                articles = response_dict["articles"]
                all_articles[option] = articles
        else:
            for option in market_options:
                response_dict = self.API.get_everything(
                    q=option.strip(),
                    language=self.configs["language"],
                    country=self.configs["country"],
                    from_param=date_start,
                    to=date_end,
                )
                articles = response_dict["articles"]
                all_articles[option] = articles

        return all_articles

    def get_category(self, market_object: dict) -> str:
        news_category = "general"
        market_category = market_object["category"]
        if market_category in self.categories:
            news_category = market_category
        return news_category

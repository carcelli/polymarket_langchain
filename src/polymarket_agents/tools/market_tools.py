import json
from pydantic import BaseModel, Field
from polymarket_agents.tooling import wrap_tool
from polymarket_agents.connectors.gamma import GammaMarketClient

gamma_client = GammaMarketClient()


class FetchActiveMarketsArgs(BaseModel):
    limit: int = Field(default=5, ge=1, le=100)


def _fetch_active_markets_impl(limit: int = 5) -> str:
    try:
        markets = gamma_client.get_markets(
            querystring_params={
                "active": True,
                "closed": False,
                "archived": False,
                "limit": limit,
            },
            parse_pydantic=True,
        )
        return json.dumps([m.model_dump() for m in markets])
    except Exception as e:
        return json.dumps([{"error": str(e)}])


fetch_active_markets = wrap_tool(
    _fetch_active_markets_impl,
    name="fetch_active_markets",
    args_schema=FetchActiveMarketsArgs,
)

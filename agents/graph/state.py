import operator
from typing import Annotated, List, TypedDict, Union, Any
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from agents.utils.objects import Market  # Importing our Pydantic model


class AgentState(TypedDict):
    """
    The cognitive state of the trader.
    It holds the conversation history, the raw market data,
    and the probabilistic reasoning derived from it.
    """

    # Messages: The reasoning history (Chat History)
    messages: Annotated[List[BaseMessage], add_messages]

    # Markets: The raw entities fetched from the "Sensorium" (Gamma API)
    # defaulting to empty list if not present
    markets: List[Market]

    # Analysis: The derived probability of the event (0.0 to 1.0)
    forecast_prob: float

    # Action: The executable instruction
    # e.g., {"side": "BUY", "size": 100.0, "token_id": "..."}
    trade_decision: dict

    # Error handling state
    error: Union[str, None]

"""
Modern Alternative: Validated Models with Pydantic v2

Industry-standard approach for validated data models.
Provides runtime validation, serialization, and error handling out-of-the-box.

Advantages over metaclasses:
- Zero boilerplate validation code
- Automatic JSON serialization/deserialization
- Rich error messages with field paths
- Type-safe by default
- Integrates with FastAPI, LangChain, etc.
- Production battle-tested

This is the recommended approach for your agent schemas, valuation models, etc.
"""

from pydantic import BaseModel, Field, field_validator, PositiveFloat, NonNegativeInt
from typing import Optional, Dict, Any
from datetime import datetime


class LineItem(BaseModel):
    """
    Line item with validated fields using Pydantic.

    This provides the same validation as the metaclass approach,
    but with modern Python data modeling patterns.
    """
    description: str = Field(..., min_length=1, description="Item description")
    weight: PositiveFloat = Field(..., description="Weight in kg")
    price: PositiveFloat = Field(..., description="Price per unit")

    @field_validator('description')
    def description_must_not_be_blank(cls, v: str) -> str:
        """Validate description is not empty or whitespace."""
        stripped = v.strip()
        if not stripped:
            raise ValueError('description cannot be empty or blank')
        return stripped

    @property
    def subtotal(self) -> float:
        """Calculate line item subtotal."""
        return self.weight * self.price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return self.model_dump()

    def to_json(self) -> str:
        """Convert to JSON string."""
        return self.model_dump_json()


# ===== ADVANCED EXAMPLES FOR YOUR USE CASES =====

class MarketDataPoint(BaseModel):
    """
    Validated market data point for time-series analysis.

    Perfect for your Polymarket data ingestion and analysis.
    """
    timestamp: datetime = Field(..., description="Data timestamp")
    symbol: str = Field(..., min_length=1, description="Market symbol")
    price: PositiveFloat = Field(..., description="Current price")
    volume: NonNegativeInt = Field(..., description="Trading volume")
    bid: Optional[PositiveFloat] = Field(None, description="Best bid price")
    ask: Optional[PositiveFloat] = Field(None, description="Best ask price")

    @field_validator('symbol')
    def symbol_must_be_uppercase(cls, v: str) -> str:
        """Ensure symbol is uppercase for consistency."""
        return v.upper()

    @field_validator('ask', 'bid')
    def ask_must_be_greater_than_bid(cls, v: Optional[float], info) -> Optional[float]:
        """Validate ask >= bid if both are present."""
        if info.field_name == 'ask' and v is not None:
            bid = info.data.get('bid')
            if bid is not None and v < bid:
                raise ValueError('ask price must be >= bid price')
        return v

    @property
    def spread(self) -> Optional[float]:
        """Calculate bid-ask spread."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None

    @property
    def mid_price(self) -> float:
        """Calculate midpoint price."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.price


class AgentPrediction(BaseModel):
    """
    Validated prediction output from ML agents.

    Ensures agent outputs conform to expected schema for downstream processing.
    """
    market_id: str = Field(..., description="Polymarket market identifier")
    predicted_probability: float = Field(..., ge=0.0, le=1.0, description="Predicted outcome probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    edge: float = Field(..., description="Predicted edge over market")
    reasoning: str = Field(..., min_length=10, description="Explanation of prediction")
    model_version: str = Field(..., description="ML model version used")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")

    @field_validator('reasoning')
    def reasoning_must_be_substantive(cls, v: str) -> str:
        """Ensure reasoning provides actual explanation."""
        if len(v.strip()) < 20:
            raise ValueError('reasoning must be at least 20 characters')
        return v.strip()

    def to_langchain_format(self) -> Dict[str, Any]:
        """Format for LangChain tool outputs."""
        return {
            "market_id": self.market_id,
            "prediction": {
                "probability": self.predicted_probability,
                "confidence": self.confidence,
                "edge": self.edge,
                "reasoning": self.reasoning
            },
            "metadata": {
                "model_version": self.model_version,
                "timestamp": self.timestamp.isoformat()
            }
        }


class ValuationModel(BaseModel):
    """
    Complex valuation model with nested validation.

    Shows how Pydantic handles complex, nested data structures
    that would be cumbersome with metaclasses.
    """
    model_name: str = Field(..., description="Valuation model identifier")
    parameters: Dict[str, float] = Field(..., description="Model parameters")
    market_data: MarketDataPoint = Field(..., description="Current market conditions")
    predictions: Dict[str, AgentPrediction] = Field(default_factory=dict,
                                                   description="Predictions by market")

    @field_validator('parameters')
    def validate_parameters(cls, v: Dict[str, float]) -> Dict[str, float]:
        """Ensure all parameters are positive."""
        for key, value in v.items():
            if value <= 0:
                raise ValueError(f'parameter {key} must be positive, got {value}')
        return v

    @field_validator('predictions')
    def validate_prediction_consistency(cls, v: Dict[str, AgentPrediction]) -> Dict[str, AgentPrediction]:
        """Ensure all predictions are for the same market."""
        if not v:
            return v

        market_ids = {pred.market_id for pred in v.values()}
        if len(market_ids) > 1:
            raise ValueError('all predictions must be for the same market')
        return v

    def compute_ensemble_prediction(self) -> float:
        """Compute weighted average of all predictions."""
        if not self.predictions:
            return 0.5  # Default neutral prediction

        total_weight = sum(pred.confidence for pred in self.predictions.values())
        if total_weight == 0:
            return 0.5

        weighted_sum = sum(pred.predicted_probability * pred.confidence
                          for pred in self.predictions.values())
        return weighted_sum / total_weight


# ===== DEMONSTRATION =====

def demo_pydantic_validation():
    """Demonstrate Pydantic-based validation."""
    print("🧮 Modern Alternative - Pydantic Validation Demo")
    print("=" * 52)

    # Basic line item
    item = LineItem(description="White mouse", weight=0.5, price=1.5)
    print(f"Item: {item.description}, weight: {item.weight}kg, price: ${item.price}")
    print(f"Subtotal: ${item.subtotal:.2f}")
    print(f"JSON: {item.to_json()}")

    # Market data point
    print("\n📈 Market Data Example:")
    market_data = MarketDataPoint(
        timestamp="2024-01-01T12:00:00Z",
        symbol="btc",
        price=45000.0,
        volume=1000,
        bid=44950.0,
        ask=45050.0
    )
    print(f"Market: {market_data.symbol} @ ${market_data.price:,.0f}")
    print(f"Spread: ${market_data.spread:.2f}, Mid: ${market_data.mid_price:.2f}")

    # Agent prediction
    print("\n🤖 Agent Prediction Example:")
    prediction = AgentPrediction(
        market_id="0x123",
        predicted_probability=0.65,
        confidence=0.8,
        edge=0.02,
        reasoning="Strong upward momentum in BTC price action over the past 24 hours",
        model_version="v2.1"
    )
    print(f"Prediction: {prediction.predicted_probability:.1%} probability")
    print(f"Confidence: {prediction.confidence:.1%}, Edge: {prediction.edge:.1%}")

    # Validation examples
    print("\n❌ Validation Examples:")
    try:
        LineItem(description="   ", weight=0.5, price=1.5)  # Blank description
    except Exception as e:
        print(f"Blank description: {e}")

    try:
        MarketDataPoint(
            timestamp="2024-01-01T12:00:00Z",
            symbol="btc",
            price=-100,  # Negative price
            volume=1000
        )
    except Exception as e:
        print(f"Negative price: {e}")

    try:
        AgentPrediction(
            market_id="0x123",
            predicted_probability=0.65,
            confidence=0.8,
            edge=0.02,
            reasoning="Too short",  # Reasoning too short
            model_version="v2.1"
        )
    except Exception as e:
        print(f"Insufficient reasoning: {e}")

    # Complex valuation model
    print("\n🏗️  Complex Valuation Model:")
    valuation = ValuationModel(
        model_name="ensemble_v1",
        parameters={"alpha": 0.1, "beta": 0.9},
        market_data=market_data,
        predictions={
            "agent1": prediction,
            "agent2": AgentPrediction(
                market_id="0x123",
                predicted_probability=0.7,
                confidence=0.6,
                edge=0.03,
                reasoning="Technical indicators show bullish divergence",
                model_version="v2.1"
            )
        }
    )
    ensemble_pred = valuation.compute_ensemble_prediction()
    print(f"Ensemble prediction: {ensemble_pred:.1%}")

    print("\n✅ Pydantic provides production-ready validation with rich error messages!")


if __name__ == "__main__":
    demo_pydantic_validation()
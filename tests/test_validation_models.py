"""
Test Validation Approaches (Fluent Python Chapter 21)

Tests for metaclass-based validation (educational) and modern alternatives.
Compares EntityMeta v7/v8 with dataclass and Pydantic approaches.
"""

import pytest
from datetime import datetime
from collections import OrderedDict

# Import metaclass examples (educational only)
from polymarket_agents.fluent_python.ch21_metaclasses.model_v7 import (
    LineItem as MetaLineItemV7,
)
from polymarket_agents.fluent_python.ch21_metaclasses.model_v8 import (
    LineItem as MetaLineItemV8,
)

# Import modern alternatives (production-ready)
from polymarket_agents.modern_alternatives.dataclass_validated import (
    LineItem as DataclassLineItem,
    TimeSeriesPoint,
)
from polymarket_agents.modern_alternatives.pydantic_validated import (
    LineItem as PydanticLineItem,
    MarketDataPoint,
    AgentPrediction,
    ValuationModel,
)


class TestMetaclassValidation:
    """Test the original metaclass approaches (educational)."""

    def test_v7_basic_validation(self):
        """Test EntityMeta v7 basic functionality."""
        item = MetaLineItemV7("White mouse", 0.5, 1.5)
        assert item.description == "White mouse"
        assert item.weight == 0.5
        assert item.price == 1.5
        assert item.subtotal() == 0.75

    def test_v7_validation_errors(self):
        """Test EntityMeta v7 validation errors."""
        with pytest.raises(ValueError, match="value must be > 0"):
            MetaLineItemV7("Mouse", -1, 1.5)

        with pytest.raises(ValueError, match="cannot be empty or blank"):
            MetaLineItemV7("", 0.5, 1.5)

    def test_v7_storage_isolation(self):
        """Test that v7 provides storage isolation."""
        item = MetaLineItemV7("Test", 1.0, 2.0)

        # Check that private storage names exist
        assert hasattr(item, "_NonBlank#description")
        assert hasattr(item, "_Quantity#weight")
        assert hasattr(item, "_Quantity#price")

        # Check that public access works
        assert item.description == "Test"
        assert item.weight == 1.0
        assert item.price == 2.0

    def test_v8_field_ordering(self):
        """Test EntityMeta v8 preserves field declaration order."""
        item = MetaLineItemV8("White mouse", 0.5, 1.5)

        # Check field order is preserved
        assert item._field_order == ("description", "weight", "price")

        # Check ordered serialization
        ordered_dict = item.to_dict()
        assert isinstance(ordered_dict, OrderedDict)
        assert list(ordered_dict.keys()) == ["description", "weight", "price"]


class TestDataclassValidation:
    """Test the dataclass-based modern alternative."""

    def test_basic_validation(self):
        """Test dataclass validation works like metaclass."""
        item = DataclassLineItem("White mouse", 0.5, 1.5)
        assert item.description == "White mouse"
        assert item.weight == 0.5
        assert item.price == 1.5
        assert item.subtotal == 0.75

    def test_validation_errors(self):
        """Test dataclass validation errors."""
        with pytest.raises(ValueError, match="value must be > 0"):
            DataclassLineItem("Mouse", -1, 1.5)

        with pytest.raises(ValueError, match="cannot be empty or blank"):
            DataclassLineItem("", 0.5, 1.5)

    def test_storage_isolation(self):
        """Test dataclass provides storage isolation like metaclass."""
        item = DataclassLineItem("Test", 1.0, 2.0)

        # Check private storage exists
        assert hasattr(item, "_NonBlank#description")
        assert hasattr(item, "_Quantity#weight")
        assert hasattr(item, "_Quantity#price")

    def test_serialization(self):
        """Test dataclass serialization."""
        item = DataclassLineItem("Mouse", 0.5, 1.5)
        data = item.to_dict()
        assert data == {"description": "Mouse", "weight": 0.5, "price": 1.5}

    def test_time_series_point(self):
        """Test the time-series example."""
        point = TimeSeriesPoint("2024-01-01T12:00:00Z", "BTC", 45000.0, 1000)
        assert point.timestamp == "2024-01-01T12:00:00Z"
        assert point.symbol == "BTC"
        assert point.price == 45000.0
        assert point.volume == 1000

        with pytest.raises(ValueError):
            TimeSeriesPoint("", "BTC", 45000.0, 1000)  # Empty timestamp


class TestPydanticValidation:
    """Test the Pydantic-based modern alternative."""

    def test_basic_validation(self):
        """Test Pydantic validation."""
        item = PydanticLineItem(description="White mouse", weight=0.5, price=1.5)
        assert item.description == "White mouse"
        assert item.weight == 0.5
        assert item.price == 1.5
        assert item.subtotal == 0.75

    def test_validation_errors(self):
        """Test Pydantic validation errors with rich messages."""
        with pytest.raises(ValueError) as exc_info:
            PydanticLineItem(description="", weight=0.5, price=1.5)
        # Pydantic v2 error message is different
        assert "String should have at least 1 character" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            PydanticLineItem(description="Mouse", weight=-1, price=1.5)
        assert "Input should be greater than 0" in str(exc_info.value)

    def test_json_serialization(self):
        """Test Pydantic JSON serialization."""
        item = PydanticLineItem(description="Mouse", weight=0.5, price=1.5)
        json_str = item.to_json()
        assert '"description":"Mouse"' in json_str
        assert '"weight":0.5' in json_str

    def test_market_data_validation(self):
        """Test market data validation."""
        data = MarketDataPoint(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="btc",
            price=45000.0,
            volume=1000,
            bid=44950.0,
            ask=45050.0,
        )

        # Symbol should be uppercased
        assert data.symbol == "BTC"

        # Spread calculation
        assert data.spread == 100.0  # 45050 - 44950
        assert data.mid_price == 45000.0  # (44950 + 45050) / 2

        # Validation: ask < bid should fail
        with pytest.raises(ValueError):
            MarketDataPoint(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                symbol="btc",
                price=45000.0,
                volume=1000,
                bid=45050.0,
                ask=44950.0,  # ask < bid
            )

    def test_agent_prediction_validation(self):
        """Test agent prediction validation."""
        prediction = AgentPrediction(
            market_id="0x123",
            predicted_probability=0.65,
            confidence=0.8,
            edge=0.02,
            reasoning="Strong upward momentum in BTC price action over the past 24 hours",
            model_version="v2.1",
        )

        assert prediction.market_id == "0x123"
        assert prediction.predicted_probability == 0.65
        assert prediction.confidence == 0.8

        # Test reasoning length validation
        with pytest.raises(ValueError):
            AgentPrediction(
                market_id="0x123",
                predicted_probability=0.65,
                confidence=0.8,
                edge=0.02,
                reasoning="Too short",  # Less than 20 chars
                model_version="v2.1",
            )

    def test_complex_valuation_model(self):
        """Test complex nested validation."""
        market_data = MarketDataPoint(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            symbol="BTC",
            price=45000.0,
            volume=1000,
        )

        prediction1 = AgentPrediction(
            market_id="market_123",
            predicted_probability=0.6,
            confidence=0.8,
            edge=0.02,
            reasoning="Technical analysis shows bullish signals",
            model_version="v1.0",
        )

        prediction2 = AgentPrediction(
            market_id="market_123",  # Same market
            predicted_probability=0.7,
            confidence=0.6,
            edge=0.03,
            reasoning="Fundamental analysis supports growth",
            model_version="v1.0",
        )

        valuation = ValuationModel(
            model_name="ensemble_v1",
            parameters={"alpha": 0.1, "beta": 0.9},
            market_data=market_data,
            predictions={"tech_agent": prediction1, "fund_agent": prediction2},
        )

        # Test ensemble prediction
        ensemble = valuation.compute_ensemble_prediction()
        expected = (0.6 * 0.8 + 0.7 * 0.6) / (0.8 + 0.6)  # Weighted average
        assert abs(ensemble - expected) < 0.001

        # Test parameter validation
        with pytest.raises(ValueError):
            ValuationModel(
                model_name="test",
                parameters={"alpha": -0.1},  # Negative parameter
                market_data=market_data,
            )

        # Test prediction consistency
        with pytest.raises(ValueError):
            ValuationModel(
                model_name="test",
                parameters={"alpha": 0.1},
                market_data=market_data,
                predictions={
                    "agent1": prediction1,
                    "agent2": AgentPrediction(  # Different market
                        market_id="different_market",
                        predicted_probability=0.5,
                        confidence=0.7,
                        edge=0.01,
                        reasoning="Different market analysis",
                        model_version="v1.0",
                    ),
                },
            )


class TestApproachComparison:
    """Compare different validation approaches."""

    def test_equivalent_functionality(self):
        """Test that all approaches provide equivalent validation."""
        test_cases = [
            ("Valid item", "Mouse", 0.5, 1.5, True),
            ("Negative weight", "Mouse", -1, 1.5, False),
            ("Empty description", "", 0.5, 1.5, False),
            ("Zero weight", "Mouse", 0, 1.5, False),
        ]

        for desc, description, weight, price, should_succeed in test_cases:
            # Test metaclass v7
            if should_succeed:
                item_v7 = MetaLineItemV7(description, weight, price)
                assert item_v7.subtotal() == weight * price
            else:
                with pytest.raises(ValueError):
                    MetaLineItemV7(description, weight, price)

            # Test dataclass
            if should_succeed:
                item_dc = DataclassLineItem(description, weight, price)
                assert item_dc.subtotal == weight * price
            else:
                with pytest.raises(ValueError):
                    DataclassLineItem(description, weight, price)

            # Test Pydantic
            if should_succeed:
                item_py = PydanticLineItem(
                    description=description, weight=weight, price=price
                )
                assert item_py.subtotal == weight * price
            else:
                with pytest.raises(ValueError):
                    PydanticLineItem(
                        description=description, weight=weight, price=price
                    )

    def test_storage_patterns(self):
        """Compare storage isolation patterns."""
        # All should provide some form of storage isolation
        meta_item = MetaLineItemV7("Test", 1.0, 2.0)
        dc_item = DataclassLineItem("Test", 1.0, 2.0)

        # Metaclass uses private attributes
        assert hasattr(meta_item, "_NonBlank#description")

        # Dataclass approach also uses private attributes
        assert hasattr(dc_item, "_NonBlank#description")

        # Pydantic uses its own internal storage
        py_item = PydanticLineItem(description="Test", weight=1.0, price=2.0)
        assert py_item.description == "Test"  # But internal storage is different


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

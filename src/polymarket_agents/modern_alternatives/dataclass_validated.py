"""
Modern Alternative: Validated Models with Dataclasses

Production-ready replacement for metaclass-based validation.
Uses dataclasses + descriptors + __post_init__ for validation.

Advantages over metaclasses:
- Readable and maintainable
- Full type checker support (mypy/pyright)
- Integrates with existing dataclass ecosystem
- No "magic" import-time behavior
- Easier testing and debugging

This achieves the same outcome as EntityMeta but with modern Python patterns.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


class Validated:
    """
    Non-data descriptor for field validation.

    Similar to the book's Validated class, but designed to work
    with dataclass fields via __post_init__.
    """

    def __init__(self, storage_name: Optional[str] = None):
        self.storage_name = storage_name

    def __get__(self, obj, objtype=None) -> Any:
        if obj is None:
            return self
        if self.storage_name:
            return getattr(obj, self.storage_name)
        raise AttributeError("Storage name not set")

    def __set__(self, obj, value):
        if self.storage_name:
            validated_value = self.validate(value)
            setattr(obj, self.storage_name, validated_value)
        else:
            raise AttributeError("Storage name not set")

    def validate(self, value):
        """Override in subclasses to implement validation."""
        return value


class Quantity(Validated):
    """A number greater than zero."""

    def validate(self, value):
        if value <= 0:
            raise ValueError("value must be > 0")
        return float(value)


class NonBlank(Validated):
    """A string with at least one non-space character."""

    def validate(self, value):
        value = value.strip()
        if not value:
            raise ValueError("value cannot be empty or blank")
        return value


@dataclass
class LineItem:
    """
    Line item with validated fields using dataclass + descriptors.

    This achieves the same validation and storage isolation as the
    metaclass approach, but with modern, readable Python.
    """

    description: str = field(default="", metadata={"validator": NonBlank()})
    weight: float = field(default=0.0, metadata={"validator": Quantity()})
    price: float = field(default=0.0, metadata={"validator": Quantity()})

    def __post_init__(self):
        """
        Apply validation after dataclass initialization.

        This mimics the import-time behavior of the metaclass,
        but happens at runtime when the object is created.
        """
        for f in self.__dataclass_fields__.values():
            validator = f.metadata.get("validator")
            if validator:
                # Get the current value (from dataclass default or __init__)
                value = getattr(self, f.name)

                # Create unique storage name (like the metaclass did)
                storage_name = f"_{type(validator).__name__}#{f.name}"
                validator.storage_name = storage_name

                # Validate and store the value
                validated_value = validator.validate(value)
                setattr(self, storage_name, validated_value)

                # Replace the field descriptor with our validator
                # This makes field access go through the descriptor
                setattr(type(self), f.name, validator)

                # Set the initial value (will go through descriptor)
                setattr(self, f.name, validated_value)

    @property
    def subtotal(self) -> float:
        """Calculate line item subtotal."""
        return self.weight * self.price

    def to_dict(self):
        """Convert to dictionary respecting field order."""
        return {
            f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()
        }


# ===== ADVANCED EXAMPLE: TIME-SERIES MODEL =====


@dataclass
class TimeSeriesPoint:
    """
    Example: Validated time-series data point for market analysis.

    Shows how this pattern extends to more complex domain models.
    """

    timestamp: str = field(metadata={"validator": NonBlank()})
    symbol: str = field(metadata={"validator": NonBlank()})
    price: float = field(default=0.0, metadata={"validator": Quantity()})
    volume: int = field(default=0, metadata={"validator": lambda x: int(max(0, x))})

    def __post_init__(self):
        # Apply validation to all fields with validators
        for f in self.__dataclass_fields__.values():
            validator = f.metadata.get("validator")
            if validator:
                value = getattr(self, f.name)
                if hasattr(validator, "validate"):
                    # Descriptor validator
                    storage_name = f"_{type(validator).__name__}#{f.name}"
                    validator.storage_name = storage_name
                    validated_value = validator.validate(value)
                    setattr(self, storage_name, validated_value)
                    setattr(type(self), f.name, validator)
                    setattr(self, f.name, validated_value)
                elif callable(validator):
                    # Simple function validator
                    validated_value = validator(value)
                    setattr(self, f.name, validated_value)


# ===== DEMONSTRATION =====


def demo_dataclass_validation():
    """Demonstrate dataclass-based validation."""
    print("🧮 Modern Alternative - Dataclass Validation Demo")
    print("=" * 55)

    # Create valid line item
    item = LineItem("White mouse", 0.5, 1.5)
    print(f"Item: {item.description}, weight: {item.weight}, price: {item.price}")
    print(f"Subtotal: ${item.subtotal:.2f}")
    print(f"As dict: {item.to_dict()}")

    # Show storage isolation (like metaclass)
    print(f"\n📦 Private storage: {item.__dict__}")

    # Test validation
    print("\n❌ Validation examples:")
    try:
        LineItem("", 0.5, 1.5)  # Empty description
    except ValueError as e:
        print(f"Empty description: {e}")

    try:
        LineItem("Mouse", -1, 1.5)  # Negative weight
    except ValueError as e:
        print(f"Negative weight: {e}")

    # Time-series example
    print("\n📈 Time-Series Example:")
    point = TimeSeriesPoint("2024-01-01T12:00:00Z", "BTC", 45000.0, 1000)
    print(f"Market data: {point.symbol} @ ${point.price:,.0f}, vol: {point.volume}")

    try:
        TimeSeriesPoint("", "BTC", 45000.0, 1000)  # Empty timestamp
    except ValueError as e:
        print(f"Empty timestamp: {e}")

    print(
        "\n✅ Dataclass validation provides same benefits with modern, readable code!"
    )


if __name__ == "__main__":
    demo_dataclass_validation()

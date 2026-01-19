"""
Fluent Python Chapter 21 - Metaclass for Automatic Descriptor Storage Names

Original book example: EntityMeta (v7)
Automatically assigns unique storage names to descriptors to avoid collisions.

WARNING: This is for educational purposes only. For production code,
prefer modern alternatives like dataclasses or Pydantic (see modern_alternatives/).

Key Concepts:
- Metaclass inspects class attributes at definition time
- Automatically renames descriptor storage to avoid conflicts
- Enables simple user code: class LineItem(Entity): ...
"""

import abc


class AutoStorage:
    """
    Descriptor that automatically gets a unique storage name.

    The EntityMeta metaclass will assign __name to each instance.
    """
    __counter = 0

    def __init__(self):
        cls = self.__class__
        prefix = cls.__name__
        index = cls.__counter
        self.storage_name = '_{}#{}'.format(prefix, index)
        cls.__counter += 1

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.storage_name)

    def __set__(self, instance, value):
        setattr(instance, self.storage_name, value)


class Validated(abc.ABC, AutoStorage):
    """Abstract base class for validated descriptors."""

    def __set__(self, instance, value):
        value = self.validate(instance, value)
        super().__set__(instance, value)

    @abc.abstractmethod
    def validate(self, instance, value):
        """Return validated value or raise ValueError."""


class Quantity(Validated):
    """A number greater than zero."""

    def validate(self, instance, value):
        if value <= 0:
            raise ValueError('value must be > 0')
        return float(value)


class NonBlank(Validated):
    """A string with at least one non-space character."""

    def validate(self, instance, value):
        value = value.strip()
        if len(value) == 0:
            raise ValueError('value cannot be empty or blank')
        return value


class EntityMeta(type):
    """
    Metaclass for business entities.

    Automatically assigns unique storage names to descriptors
    by calling __set_name__ on each descriptor found in the class.
    """

    def __new__(meta_cls, name, bases, namespace, **kwargs):
        for key, attr in namespace.items():
            if isinstance(attr, Validated):
                attr.storage_name = f'_{attr.__class__.__name__}#{key}'
        return super().__new__(meta_cls, name, bases, namespace, **kwargs)


class Entity(metaclass=EntityMeta):
    """Business entity base class with automatic descriptor naming."""


# ===== EXAMPLE USAGE =====

class LineItem(Entity):
    """Line item with validated fields."""
    description = NonBlank()
    weight = Quantity()
    price = Quantity()

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price


# ===== DEMONSTRATION =====

def demo_v7():
    """Demonstrate the v7 metaclass approach."""
    print("🧮 Fluent Python Ch21 - EntityMeta v7 Demo")
    print("=" * 50)

    # Create a valid line item
    item = LineItem("White mouse", 0.5, 1.5)
    print(f"Item: {item.description}, weight: {item.weight}, price: {item.price}")
    print(f"Subtotal: {item.subtotal()}")

    # Show storage names are unique
    print("\n📦 Storage names:")
    print(f"description -> {item.__dict__}")
    print(f"weight -> {item.__dict__}")
    print(f"price -> {item.__dict__}")

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

    print("\n✅ Metaclass automatically assigned unique storage names!")


if __name__ == "__main__":
    demo_v7()
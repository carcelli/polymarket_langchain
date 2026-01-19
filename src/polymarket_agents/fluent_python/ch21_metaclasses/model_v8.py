"""
Fluent Python Chapter 21 - Metaclass with Field Ordering (v8)

Enhanced EntityMeta with __prepare__ to preserve field declaration order.

WARNING: This is for educational purposes only. For production code,
prefer modern alternatives like dataclasses or Pydantic.

Key Enhancement:
- Uses __prepare__ to return OrderedDict, preserving field order
- Maintains insertion order for serialization/validation sequences
"""

import abc
from collections import OrderedDict


class AutoStorage:
    """Descriptor with automatic storage name assignment."""
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
    Metaclass for business entities with field ordering.

    Uses __prepare__ to return OrderedDict, preserving the order
    in which fields are declared in the class body.
    """

    @classmethod
    def __prepare__(mcs, name, bases):
        """Return OrderedDict to preserve field declaration order."""
        return OrderedDict()

    def __new__(meta_cls, name, bases, namespace, **kwargs):
        # Get field declaration order from OrderedDict keys
        fields = [key for key in namespace.keys()
                 if isinstance(namespace[key], Validated)]

        # Store field order for potential use in serialization/validation
        namespace['_field_order'] = tuple(fields)

        # Assign storage names to descriptors
        for key, attr in namespace.items():
            if isinstance(attr, Validated):
                attr.storage_name = f'_{attr.__class__.__name__}#{key}'

        return super().__new__(meta_cls, name, bases, namespace, **kwargs)


class Entity(metaclass=EntityMeta):
    """Business entity base class with field ordering."""
    _field_order = ()  # Default empty tuple


# ===== EXAMPLE USAGE =====

class LineItem(Entity):
    """Line item with validated fields in specific order."""
    description = NonBlank()  # First field
    weight = Quantity()       # Second field
    price = Quantity()        # Third field

    def __init__(self, description, weight, price):
        self.description = description
        self.weight = weight
        self.price = price

    def subtotal(self):
        return self.weight * self.price

    def to_dict(self):
        """Serialize respecting field order."""
        return OrderedDict(
            (field, getattr(self, field))
            for field in self._field_order
        )


# ===== DEMONSTRATION =====

def demo_v8():
    """Demonstrate the v8 metaclass with field ordering."""
    print("🧮 Fluent Python Ch21 - EntityMeta v8 Demo (with ordering)")
    print("=" * 60)

    # Create a line item
    item = LineItem("White mouse", 0.5, 1.5)
    print(f"Item: {item.description}, weight: {item.weight}, price: {item.price}")
    print(f"Subtotal: {item.subtotal()}")

    # Show field order is preserved
    print(f"\n📋 Field declaration order: {item._field_order}")

    # Show ordered serialization
    print(f"Ordered dict: {item.to_dict()}")

    # Show storage names (unique per field)
    print("\n📦 Storage details:")
    for field in item._field_order:
        descriptor = getattr(type(item), field)
        print(f"{field}: storage_name='{descriptor.storage_name}', value={getattr(item, field)}")

    # Test validation with ordered error reporting
    print("\n❌ Validation examples:")
    try:
        LineItem("", 0.5, 1.5)  # Empty description (first field)
    except ValueError as e:
        print(f"Empty description: {e}")

    try:
        LineItem("Mouse", -1, 1.5)  # Negative weight (second field)
    except ValueError as e:
        print(f"Negative weight: {e}")

    print("\n✅ Metaclass preserved field order and assigned unique storage names!")


if __name__ == "__main__":
    demo_v8()
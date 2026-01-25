"""
Vector2d - Pythonic Object Design (Fluent Python Chapter 9)

Demonstrates key concepts from Chapter 9:
- Object representations (__repr__, __str__, __format__, __bytes__)
- Alternative constructors (@classmethod)
- Read-only properties (@property)
- Hashable objects (__hash__, __eq__)
- Memory optimization (__slots__)

This example shows how to create objects that behave like built-in types,
providing multiple representations and supporting various Python protocols.
"""

from array import array
import math
from typing import Iterable


class Vector2d:
    """
    A two-dimensional vector class demonstrating Pythonic object design.

    Features from Chapter 9:
    - Multiple string representations (__repr__, __str__, __format__)
    - Binary representation (__bytes__)
    - Alternative constructor (frombytes classmethod)
    - Read-only properties for x, y components
    - Hashable for use in sets/dicts (__hash__, __eq__)
    - Memory-efficient with __slots__
    """

    __slots__ = ("__x", "__y")  # Memory optimization (Chapter 9)
    typecode = "d"

    def __init__(self, x, y):
        """Initialize with float conversion for type safety."""
        self.__x = float(x)
        self.__y = float(y)

    @property
    def x(self):
        """Read-only x component (Chapter 9)."""
        return self.__x

    @property
    def y(self):
        """Read-only y component (Chapter 9)."""
        return self.__y

    def __iter__(self):
        """Make vector iterable for unpacking (x, y = vector)."""
        return (i for i in (self.x, self.y))

    def __repr__(self):
        """Developer-friendly representation (Chapter 9)."""
        class_name = type(self).__name__
        return "{}({!r}, {!r})".format(class_name, *self)

    def __str__(self):
        """User-friendly string representation (Chapter 9)."""
        return str(tuple(self))

    def __bytes__(self):
        """Binary representation with typecode prefix (Chapter 9)."""
        return bytes([ord(self.typecode)]) + bytes(array(self.typecode, self))

    def __eq__(self, other):
        """Equality comparison for hashing support (Chapter 9)."""
        return tuple(self) == tuple(other)

    def __hash__(self):
        """Hash support using XOR of component hashes (Chapter 9)."""
        return hash(self.x) ^ hash(self.y)

    def __abs__(self):
        """Magnitude of the vector."""
        return math.hypot(self.x, self.y)

    def __bool__(self):
        """Non-zero vectors are truthy."""
        return bool(abs(self))

    def __format__(self, fmt_spec=""):
        """Custom format specification with polar coordinates (Chapter 9)."""
        if fmt_spec.endswith("p"):
            # Polar coordinates: <magnitude, angle>
            fmt_spec = fmt_spec[:-1]
            coords = (abs(self), math.atan2(self.y, self.x))
            outer_fmt = "<{}, {}>"
        else:
            # Cartesian coordinates: (x, y)
            coords = self
            outer_fmt = "({}, {})"

        components = (format(c, fmt_spec) for c in coords)
        return outer_fmt.format(*components)

    @classmethod
    def frombytes(cls, octets):
        """Alternative constructor from binary data (Chapter 9)."""
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(*memv)

    def angle(self):
        """Angle in radians for polar coordinates."""
        return math.atan2(self.y, self.x)


# Demonstration of Chapter 9 concepts
if __name__ == "__main__":
    print("🧮 Vector2d - Pythonic Object Design Demo")
    print("=" * 50)

    # Basic usage
    v1 = Vector2d(3, 4)
    print(f"Vector: {v1}")
    print(f"Components: x={v1.x}, y={v1.y}")
    print(f"Magnitude: {abs(v1)}")
    print(f"Boolean: {bool(v1)}")

    # Unpacking (iterable)
    x, y = v1
    print(f"Unpacked: x={x}, y={y}")

    # Representations
    print("\n📝 Representations:")
    print(f"  repr(): {repr(v1)}")
    print(f"  str(): {str(v1)}")
    print(f"  format(): {format(v1)}")
    print(f"  format('.2f'): {format(v1, '.2f')}")
    print(f"  format('p'): {format(v1, 'p')}")  # Polar

    # Binary representation and alternative constructor
    print("\n💾 Binary serialization:")
    binary = bytes(v1)
    print(f"  bytes(): {binary}")
    v2 = Vector2d.frombytes(binary)
    print(f"  reconstructed: {v2}")
    print(f"  equal: {v1 == v2}")

    # Hashable (can be used in sets and as dict keys)
    print("\n🏷️  Hashable:")
    print(f"  hash(v1): {hash(v1)}")
    v3 = Vector2d(3, 4)  # Equal to v1
    print(f"  v1 == v3: {v1 == v3}")
    print(f"  hash(v1) == hash(v3): {hash(v1) == hash(v3)}")
    print(f"  in set: {v1 in {v2, v3}}")

    print("\n✅ All Chapter 9 concepts demonstrated!")

"""
Enhanced Vector2d - Operator Overloading Best Practices (Fluent Python Chapter 13)

Combines Chapter 9 (Pythonic Object Design) with Chapter 13 (Operator Overloading):
- Object representations (__repr__, __str__, __format__, __bytes__)
- Alternative constructors (@classmethod)
- Read-only properties (@property)
- Hashable objects (__hash__, __eq__)
- Memory optimization (__slots__)

NEW in Chapter 13:
- Arithmetic operators (+, *, @ for dot product)
- Proper NotImplemented handling for unsupported operands
- Reverse operators (__radd__, __rmul__, __rmatmul__)
- Mixed-type operations with graceful fallbacks
- Immutable behavior (no in-place mutation)

Perfect for domain objects in financial/ML contexts:
- Position combining: pos1 + pos2
- Signal scaling: signal * confidence
- Edge calculation: forecast @ market_probs
- Fast lookups: signal_cache[signal] = analysis
"""

from array import array
import math
from typing import Any, Iterable
import numbers
import itertools


class Vector2d:
    """
    Enhanced two-dimensional vector with full operator overloading support.

    Chapter 9 features:
    - Multiple string representations (__repr__, __str__, __format__)
    - Binary representation (__bytes__)
    - Alternative constructor (frombytes classmethod)
    - Read-only properties for x, y components
    - Hashable for use in sets/dicts (__hash__, __eq__)
    - Memory-efficient with __slots__

    Chapter 13 additions:
    - Arithmetic operators: +, *, @ (dot product)
    - Proper NotImplemented handling for type compatibility
    - Reverse operators for commutativity
    - Mixed-type operations with iterables
    """

    __slots__ = ("__x", "__y")  # Memory optimization + immutability guarantee
    typecode = "d"

    def __init__(self, x: float, y: float):
        """Initialize with float conversion for type safety."""
        self.__x = float(x)
        self.__y = float(y)

    @property
    def x(self) -> float:
        """Read-only x component."""
        return self.__x

    @property
    def y(self) -> float:
        """Read-only y component."""
        return self.__y

    def __iter__(self):
        """Make vector iterable for unpacking."""
        return iter((self.x, self.y))

    def __repr__(self):
        """Developer-friendly representation."""
        class_name = type(self).__name__
        return "{}({!r}, {!r})".format(class_name, *self)

    def __str__(self):
        """User-friendly string representation."""
        return str(tuple(self))

    def __bytes__(self):
        """Binary representation with typecode prefix."""
        return bytes([ord(self.typecode)]) + bytes(array(self.typecode, self))

    def __eq__(self, other: Any) -> bool:
        """Equality comparison - return NotImplemented for unsupported types."""
        if isinstance(other, Vector2d):
            return (self.x, self.y) == (other.x, other.y)
        return NotImplemented

    def __hash__(self):
        """Hash support using XOR of component hashes."""
        return hash(self.x) ^ hash(self.y)

    def __abs__(self):
        """Magnitude of the vector."""
        return math.hypot(self.x, self.y)

    def __bool__(self):
        """Non-zero vectors are truthy."""
        return bool(abs(self))

    # ===== ARITHMETIC OPERATORS (Chapter 13) =====

    def __add__(self, other: Any):
        """Vector addition - supports Vector2d and iterables."""
        if isinstance(other, Vector2d):
            return Vector2d(self.x + other.x, self.y + other.y)
        try:
            # Support adding to iterables (e.g., tuple, list)
            x, y = other
            return Vector2d(self.x + x, self.y + y)
        except (TypeError, ValueError):
            return NotImplemented

    __radd__ = __add__  # + is commutative

    def __mul__(self, scalar: Any):
        """Scalar multiplication."""
        if isinstance(scalar, numbers.Real):
            return Vector2d(self.x * scalar, self.y * scalar)
        return NotImplemented

    __rmul__ = __mul__  # * is commutative

    def __matmul__(self, other: Any):
        """Dot product using @ operator (Python 3.5+)."""
        if isinstance(other, Vector2d):
            return self.x * other.x + self.y * other.y
        return NotImplemented

    __rmatmul__ = __matmul__  # @ is commutative for dot product

    def __format__(self, fmt_spec: str = "") -> str:
        """Custom format specification with polar coordinates."""
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
    def frombytes(cls, octets: bytes):
        """Alternative constructor from binary data."""
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(*memv)

    def angle(self) -> float:
        """Angle in radians for polar coordinates."""
        return math.atan2(self.y, self.x)


# ===== DEMONSTRATION OF OPERATOR OVERLOADING =====


def demo_operator_overloading():
    """Demonstrates Chapter 13 operator overloading concepts."""
    print("🧮 Enhanced Vector2d - Operator Overloading Demo")
    print("=" * 60)

    # Basic vectors
    v1 = Vector2d(3, 4)
    v2 = Vector2d(1, 2)
    print(f"v1 = {v1}")
    print(f"v2 = {v2}")

    # Vector addition
    print("\n➕ Vector Addition:")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"v2 + v1 = {v2 + v1}")  # Commutative

    # Mixed-type addition
    print("\n🔄 Mixed-Type Addition:")
    print(f"v1 + (10, 20) = {v1 + (10, 20)}")
    print(f"(10, 20) + v1 = {(10, 20) + v1}")  # Reverse operator

    # Scalar multiplication
    print("\n✖️ Scalar Multiplication:")
    print(f"v1 * 3 = {v1 * 3}")
    print(f"3 * v1 = {3 * v1}")  # Commutative
    print(f"v1 * 0.5 = {v1 * 0.5}")

    # Dot product
    print("\n🔺 Dot Product (@ operator):")
    print(f"v1 @ v2 = {v1 @ v2}")
    print(f"v2 @ v1 = {v2 @ v1}")  # Commutative

    # Error handling
    print("\n❌ Error Handling (returns NotImplemented):")
    try:
        result = v1 + "invalid"
        print(f"v1 + 'invalid' = {result}")
    except TypeError as e:
        print(f"v1 + 'invalid' → TypeError: {e}")

    # Hashing and equality
    print("\n🏷️ Hashing & Equality:")
    v3 = Vector2d(3, 4)  # Equal to v1
    print(f"v1 == v3: {v1 == v3}")
    print(f"hash(v1) == hash(v3): {hash(v1) == hash(v3)}")
    print(f"v1 in {{v2, v3}}: {v1 in {v2, v3}}")

    # Formatting
    print("\n📝 Formatting:")
    print(f"format(v1, '.2f'): {format(v1, '.2f')}")
    print(f"format(v1, 'p'): {format(v1, 'p')}")  # Polar coordinates

    print("\n✅ All Chapter 13 operator overloading concepts demonstrated!")


if __name__ == "__main__":
    demo_operator_overloading()

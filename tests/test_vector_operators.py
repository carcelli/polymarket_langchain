"""
Test Enhanced Vector2d Operator Overloading (Chapter 13)

Tests all operator overloading features:
- Arithmetic operators (+, *, @)
- Reverse operators (__radd__, __rmul__, __rmatmul__)
- Mixed-type operations
- NotImplemented handling for unsupported operands
- Edge cases and error conditions
"""

import pytest
import math
from polymarket_agents.utils.vector import Vector2d


class TestVectorArithmetic:
    """Test arithmetic operators from Chapter 13."""

    def test_vector_addition(self):
        """Test vector + vector addition."""
        v1 = Vector2d(1, 2)
        v2 = Vector2d(3, 4)
        result = v1 + v2
        assert result == Vector2d(4, 6)
        assert isinstance(result, Vector2d)

    def test_vector_addition_commutative(self):
        """Test commutativity: v1 + v2 == v2 + v1."""
        v1 = Vector2d(1, 2)
        v2 = Vector2d(3, 4)
        assert v1 + v2 == v2 + v1

    def test_vector_scalar_multiplication(self):
        """Test vector * scalar multiplication."""
        v = Vector2d(2, 3)
        result = v * 3
        assert result == Vector2d(6, 9)
        assert isinstance(result, Vector2d)

    def test_scalar_vector_multiplication_commutative(self):
        """Test commutativity: v * s == s * v."""
        v = Vector2d(2, 3)
        scalar = 3
        assert v * scalar == scalar * v

    def test_vector_dot_product(self):
        """Test vector @ vector dot product."""
        v1 = Vector2d(1, 2)
        v2 = Vector2d(3, 4)
        result = v1 @ v2
        assert result == (1*3 + 2*4)  # 3 + 8 = 11
        assert isinstance(result, (int, float))

    def test_dot_product_commutative(self):
        """Test commutativity of dot product."""
        v1 = Vector2d(1, 2)
        v2 = Vector2d(3, 4)
        assert v1 @ v2 == v2 @ v1


class TestMixedTypeOperations:
    """Test operations between Vector2d and other types."""

    def test_vector_tuple_addition(self):
        """Test Vector2d + tuple."""
        v = Vector2d(1, 2)
        t = (3, 4)
        result = v + t
        assert result == Vector2d(4, 6)

    def test_tuple_vector_addition_reverse(self):
        """Test tuple + Vector2d (reverse operator)."""
        v = Vector2d(1, 2)
        t = (3, 4)
        result = t + v  # Uses __radd__
        assert result == Vector2d(4, 6)

    def test_vector_list_addition(self):
        """Test Vector2d + list."""
        v = Vector2d(1, 2)
        lst = [3, 4]
        result = v + lst
        assert result == Vector2d(4, 6)

    def test_vector_int_multiplication(self):
        """Test Vector2d * int."""
        v = Vector2d(2, 3)
        result = v * 5
        assert result == Vector2d(10, 15)

    def test_vector_float_multiplication(self):
        """Test Vector2d * float."""
        v = Vector2d(2, 3)
        result = v * 2.5
        assert result == Vector2d(5.0, 7.5)


class TestNotImplementedHandling:
    """Test proper NotImplemented handling for unsupported operations."""

    def test_vector_string_addition_returns_notimplemented(self):
        """Test Vector2d + str returns NotImplemented."""
        v = Vector2d(1, 2)
        result = v.__add__("invalid")
        assert result is NotImplemented

    def test_vector_string_multiplication_returns_notimplemented(self):
        """Test Vector2d * str returns NotImplemented."""
        v = Vector2d(1, 2)
        result = v.__mul__("invalid")
        assert result is NotImplemented

    def test_vector_string_dot_product_returns_notimplemented(self):
        """Test Vector2d @ str returns NotImplemented."""
        v = Vector2d(1, 2)
        result = v.__matmul__("invalid")
        assert result is NotImplemented

    def test_unsupported_addition_raises_typeerror(self):
        """Test that unsupported addition raises TypeError."""
        v = Vector2d(1, 2)
        with pytest.raises(TypeError):
            result = v + "invalid"

    def test_unsupported_multiplication_raises_typeerror(self):
        """Test that unsupported multiplication raises TypeError."""
        v = Vector2d(1, 2)
        with pytest.raises(TypeError):
            result = v * "invalid"


class TestEdgeCases:
    """Test edge cases and special conditions."""

    def test_addition_with_wrong_length_tuple(self):
        """Test addition with tuple of wrong length."""
        v = Vector2d(1, 2)
        with pytest.raises(TypeError):
            result = v + (1, 2, 3)  # Too many elements

    def test_addition_with_non_iterable(self):
        """Test addition with non-iterable."""
        v = Vector2d(1, 2)
        with pytest.raises(TypeError):
            result = v + 42  # Not iterable

    def test_multiplication_with_complex_number(self):
        """Test multiplication with complex number (not a Real)."""
        v = Vector2d(1, 2)
        complex_num = 3 + 4j
        with pytest.raises(TypeError):
            result = v * complex_num

    def test_zero_vector_operations(self):
        """Test operations with zero vectors."""
        zero = Vector2d(0, 0)
        v = Vector2d(1, 2)

        # Addition
        assert zero + v == v
        assert v + zero == v

        # Scalar multiplication
        assert zero * 5 == Vector2d(0, 0)
        assert v * 0 == Vector2d(0, 0)

        # Dot product
        assert zero @ v == 0
        assert v @ zero == 0


class TestImmutability:
    """Test that operations create new instances (immutability)."""

    def test_addition_creates_new_instance(self):
        """Test that + creates a new Vector2d instance."""
        v1 = Vector2d(1, 2)
        v2 = Vector2d(3, 4)
        result = v1 + v2

        assert result is not v1
        assert result is not v2
        assert v1 == Vector2d(1, 2)  # Original unchanged
        assert v2 == Vector2d(3, 4)  # Original unchanged

    def test_multiplication_creates_new_instance(self):
        """Test that * creates a new Vector2d instance."""
        v = Vector2d(1, 2)
        result = v * 3

        assert result is not v
        assert v == Vector2d(1, 2)  # Original unchanged


class TestHashingAndEquality:
    """Test that operator overloading preserves hashing behavior."""

    def test_equal_vectors_have_same_hash(self):
        """Test that equal vectors have equal hashes."""
        v1 = Vector2d(3, 4)
        v2 = Vector2d(3, 4)
        assert v1 == v2
        assert hash(v1) == hash(v2)

    def test_vectors_usable_in_sets(self):
        """Test that vectors can be used in sets."""
        v1 = Vector2d(1, 2)
        v2 = Vector2d(3, 4)
        v3 = Vector2d(1, 2)  # Equal to v1

        s = {v1, v2, v3}
        assert len(s) == 2  # v1 and v3 should be considered the same
        assert v1 in s
        assert v2 in s


class TestFormatting:
    """Test that formatting still works with operators."""

    def test_polar_formatting_with_operations(self):
        """Test polar formatting on computed vectors."""
        v1 = Vector2d(3, 4)
        v2 = Vector2d(1, 2)
        result = v1 + v2

        # Should work with polar formatting
        polar_str = format(result, 'p')
        assert '<' in polar_str and '>' in polar_str


if __name__ == "__main__":
    pytest.main([__file__])
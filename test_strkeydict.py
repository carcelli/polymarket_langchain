#!/usr/bin/env python3
"""
Test script to verify StrKeyDict functionality.
"""

from src.polymarket_agents.utils.structures import StrKeyDict

def test_robustness():
    """Test the StrKeyDict's cross-type key handling."""
    print("Testing StrKeyDict robustness...")

    # 1. Initialize
    d = StrKeyDict()

    # 2. Insert with Mixed Types
    d[123] = "Integer Key"
    d["456"] = "String Key"

    # 3. Verify Cross-Type Lookup
    assert d["123"] == "Integer Key"  # Look up int with str
    assert d[456] == "String Key"     # Look up str with int

    # 4. Verify Existence
    assert 123 in d
    assert "123" in d

    # 5. Test Missing Key Handling
    try:
        _ = d["nonexistent"]
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

    # 6. Test Get Method
    assert d.get(123) == "Integer Key"
    assert d.get("nonexistent", "default") == "default"

    # 7. Test Update Method
    d.update({789: "Updated Key"})
    assert d[789] == "Updated Key"
    assert d["789"] == "Updated Key"

    print("✅ StrKeyDict is working correctly!")

if __name__ == "__main__":
    test_robustness()
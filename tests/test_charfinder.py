"""
Test Character Finder Server and Client (Fluent Python Chapter 16)

Tests the asyncio TCP character finder server and its client integration.
These tests require the server to be running on localhost:2323.

To run the server for testing:
    PYTHONPATH=src python -m polymarket_agents.utils.charfinder_server 2323

Then run these tests:
    PYTHONPATH=src python -m pytest tests/test_charfinder.py -v
"""

import pytest
import asyncio
import socket
from typing import List

from polymarket_agents.utils.charfinder_client import (
    query_unicode_names,
    async_query_unicode_names,
    CharacterFinderClient,
)


class TestCharacterFinderClient:
    """Test the synchronous character finder client."""

    def test_client_initialization(self):
        """Test client initialization with default parameters."""
        client = CharacterFinderClient()
        assert client.host == "127.0.0.1"
        assert client.port == 2323
        assert client.timeout == 5.0

    def test_client_custom_params(self):
        """Test client initialization with custom parameters."""
        client = CharacterFinderClient(host="localhost", port=9999, timeout=10.0)
        assert client.host == "localhost"
        assert client.port == 9999
        assert client.timeout == 10.0

    def test_unicode_search_basic(self):
        """Test basic Unicode character search."""
        try:
            results = query_unicode_names("chess")
            assert isinstance(results, list)

            # Should find some chess-related characters
            assert len(results) > 0

            # Check result format
            for result in results:
                assert result.startswith("U+")
                assert "\t" in result  # Tab-separated format

        except (ConnectionError, TimeoutError):
            pytest.skip(
                "Character finder server not running - start with: python -m polymarket_agents.utils.charfinder_server"
            )

    def test_unicode_search_arrows(self):
        """Test searching for arrow symbols."""
        try:
            results = query_unicode_names("arrow")
            assert isinstance(results, list)
            assert len(results) > 0

            # Should contain various arrow symbols
            arrow_found = any("ARROW" in result.upper() for result in results)
            assert arrow_found, "Should find arrow symbols"

        except (ConnectionError, TimeoutError):
            pytest.skip("Character finder server not running")

    def test_unicode_search_no_results(self):
        """Test search that returns no results."""
        try:
            results = query_unicode_names("nonexistent_search_term_12345")
            assert isinstance(results, list)
            assert len(results) == 0

        except (ConnectionError, TimeoutError):
            pytest.skip("Character finder server not running")

    def test_connection_error_handling(self):
        """Test handling of connection errors."""
        # Try to connect to a non-existent server
        client = CharacterFinderClient(host="127.0.0.1", port=60000, timeout=1.0)

        with pytest.raises(ConnectionError):
            client.query("test")


class TestAsyncCharacterFinderClient:
    """Test the asynchronous character finder client."""

    @pytest.mark.asyncio
    async def test_async_unicode_search(self):
        """Test asynchronous Unicode character search."""
        try:
            results = await async_query_unicode_names("greek")
            assert isinstance(results, list)
            assert len(results) > 0

            # Should find Greek alphabet characters
            greek_found = any("GREEK" in result.upper() for result in results)
            assert greek_found, "Should find Greek alphabet symbols"

        except (ConnectionError, TimeoutError):
            pytest.skip("Character finder server not running")

    @pytest.mark.asyncio
    async def test_async_connection_error(self):
        """Test async handling of connection errors."""
        with pytest.raises(ConnectionError):
            await async_query_unicode_names(
                "test", host="127.0.0.1", port=60000, timeout=1.0
            )


class TestServerIntegration:
    """Test integration with the character finder server."""

    def test_server_availability(self):
        """Test if the character finder server is running."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)

        try:
            result = sock.connect_ex(("127.0.0.1", 2323))
            sock.close()

            if result != 0:
                pytest.skip(
                    "Character finder server not running on localhost:2323 - start with: python -m polymarket_agents.utils.charfinder_server"
                )

        except Exception:
            pytest.skip("Cannot test server availability")


def test_convenience_function():
    """Test the synchronous convenience function."""
    try:
        results = query_unicode_names("smiley")
        assert isinstance(results, list)

        # Should find smiley face characters
        smiley_found = any(
            "SMILEY" in result.upper() or "FACE" in result.upper() for result in results
        )
        assert smiley_found, "Should find smiley face symbols"

    except (ConnectionError, TimeoutError):
        pytest.skip("Character finder server not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

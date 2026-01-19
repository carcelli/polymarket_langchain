"""
Character Finder Client (Fluent Python Chapter 16 Integration)

Asyncio client for querying the TCP character finder server.
Provides synchronous and asynchronous interfaces for agent integration.

Usage in agents:
    # Synchronous (blocking)
    from polymarket_agents.utils.charfinder_client import query_unicode_names
    results = query_unicode_names("chess")
    # Returns: ['U+2654\t♔\tWHITE CHESS KING', ...]

    # Asynchronous (non-blocking)
    import asyncio
    from polymarket_agents.utils.charfinder_client import async_query_unicode_names

    async def agent_task():
        results = await async_query_unicode_names("arrow")
        return results
"""

import asyncio
import socket
from typing import List, Optional


class CharacterFinderClient:
    """
    Synchronous client for the Unicode character finder server.
    """

    def __init__(self, host: str = '127.0.0.1', port: int = 2323, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout

    def query(self, search_term: str) -> List[str]:
        """
        Query the character finder server for Unicode names containing search_term.

        Args:
            search_term: String to search for in Unicode character names

        Returns:
            List of result strings in format: "U+XXXX\tCHAR\tNAME"

        Raises:
            ConnectionError: If cannot connect to server
            TimeoutError: If query times out
        """
        try:
            # Create socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect((self.host, self.port))

            # Read the prompt
            prompt = sock.recv(1024)
            if not prompt.endswith(b'?> '):
                sock.close()
                raise ConnectionError("Did not receive expected prompt")

            # Send the query
            sock.sendall(search_term.encode('utf-8') + b'\n')

            # Read response
            response = b''
            sock.settimeout(1.0)  # Shorter timeout for reading

            while True:
                try:
                    chunk = sock.recv(1024)
                    if not chunk:
                        break
                    response += chunk
                    if response.endswith(b'?> '):
                        # Found next prompt, extract response
                        response = response[:-3]  # Remove prompt
                        break
                except socket.timeout:
                    break

            sock.close()

            # Parse response into lines
            if response:
                text = response.decode('utf-8', errors='replace')
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                # Filter out status messages, keep only result lines
                results = [line for line in lines if line.startswith('U+')]
                return results
            else:
                return []

        except socket.timeout as e:
            raise TimeoutError(f"Query timed out: {e}") from e
        except (ConnectionRefusedError, socket.gaierror) as e:
            raise ConnectionError(f"Cannot connect to server {self.host}:{self.port}: {e}") from e


async def async_query_unicode_names(search_term: str,
                                  host: str = '127.0.0.1',
                                  port: int = 2323,
                                  timeout: float = 5.0) -> List[str]:
    """
    Asynchronous query of the Unicode character finder server.

    Args:
        search_term: String to search for in Unicode character names
        host: Server hostname or IP
        port: Server port number
        timeout: Connection timeout in seconds

    Returns:
        List of result strings in format: "U+XXXX\tCHAR\tNAME"
    """
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=timeout
        )

        # Read prompt
        prompt = await asyncio.wait_for(reader.readuntil(b'?> '), timeout=timeout)
        if not prompt.endswith(b'?> '):
            writer.close()
            await writer.wait_closed()
            raise ConnectionError("Did not receive expected prompt")

        # Send query
        writer.write(search_term.encode('utf-8') + b'\n')
        await writer.drain()

        # Read response
        response = await asyncio.wait_for(reader.readuntil(b'?> '), timeout=timeout)

        writer.close()
        await writer.wait_closed()

        # Parse response
        if response:
            text = response.decode('utf-8', errors='replace')
            # Remove the final prompt
            if text.endswith('?> '):
                text = text[:-3]

            lines = [line.strip() for line in text.split('\n') if line.strip()]
            # Filter out status messages, keep only result lines
            results = [line for line in lines if line.startswith('U+')]
            return results

        return []

    except asyncio.TimeoutError:
        raise TimeoutError(f"Query timed out after {timeout}s")
    except (ConnectionRefusedError, OSError) as e:
        raise ConnectionError(f"Cannot connect to server {host}:{port}: {e}") from e


# Synchronous convenience function
def query_unicode_names(search_term: str,
                       host: str = '127.0.0.1',
                       port: int = 2323,
                       timeout: float = 5.0) -> List[str]:
    """
    Synchronous convenience function for Unicode name queries.

    Perfect for use in regular (non-async) agent code.
    """
    client = CharacterFinderClient(host, port, timeout)
    return client.query(search_term)


# Demo and testing functions
def demo_sync_client():
    """Demonstrate synchronous client usage."""
    print("🔍 Character Finder Client Demo")
    print("=" * 40)

    client = CharacterFinderClient()

    test_queries = ["chess", "arrow", "greek", "smiley", "nonexistent"]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            results = client.query(query)
            print(f"Found {len(results)} matches:")
            for result in results[:3]:  # Show first 3 results
                print(f"  {result}")
            if len(results) > 3:
                print(f"  ... and {len(results) - 3} more")
        except Exception as e:
            print(f"Error: {e}")


async def demo_async_client():
    """Demonstrate asynchronous client usage."""
    print("\n🔍 Async Character Finder Client Demo")
    print("=" * 40)

    test_queries = ["chess", "arrow", "greek"]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        try:
            results = await async_query_unicode_names(query)
            print(f"Found {len(results)} matches:")
            for result in results[:2]:  # Show first 2 results
                print(f"  {result}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == '__main__':
    # Run sync demo
    demo_sync_client()

    # Run async demo
    asyncio.run(demo_async_client())
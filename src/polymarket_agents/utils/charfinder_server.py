"""
Asyncio TCP Character Finder Server (Fluent Python Chapter 16)

Provides a coroutine-based TCP service for Unicode character name searches.
Agents can query this internal microservice for symbol → name lookups,
metadata enrichment, or ontology mapping.

Usage:
    python -m polymarket_agents.utils.charfinder_server [port]

Example queries:
    - "chess" → finds chess-related Unicode symbols
    - "arrow" → finds various arrow symbols
    - "greek" → finds Greek alphabet symbols

Technical Details:
- Single-threaded asyncio server (no GIL issues)
- Each client connection is a coroutine
- Efficient Unicode name index with persistent caching
- Telnet-compatible protocol for easy testing
"""

import asyncio
import socket
import sys
import pickle
import unicodedata
from collections import namedtuple
from typing import List, Iterator, Optional


# Index file for caching
INDEX_FILE = 'charfinder_index.pickle'

# Protocol constants
CRLF = b'\r\n'
PROMPT = b'?> '

# Named tuple for search results
Result = namedtuple('Result', 'code char name')


class UnicodeNameIndex:
    """
    Builds and queries an index of Unicode character names.

    Creates a searchable index mapping words to Unicode characters.
    Persists the index to disk for fast subsequent loads.
    """

    def __init__(self, chars: Optional[str] = None, pickle_file: str = INDEX_FILE):
        self.pickle_file = pickle_file
        self.index = None
        if chars is None:
            chars = ''.join(chr(i) for i in range(32, 0x10FFFF + 1))
        self.load(chars)

    def load(self, chars: str) -> None:
        """
        Load or build the character index.

        Tries to load from pickle file first, builds from scratch if needed.
        """
        try:
            with open(self.pickle_file, 'rb') as fp:
                self.index = pickle.load(fp)
        except (FileNotFoundError, EOFError, pickle.PickleError):
            # Build index from scratch
            self.index = {}
            print(f'Building {len(chars)} character index...')
            for char in chars:
                try:
                    name = unicodedata.name(char)
                    if name:
                        self.index[char] = name
                except ValueError:
                    pass  # Skip characters without names

            # Save for future use
            with open(self.pickle_file, 'wb') as fp:
                pickle.dump(self.index, fp)
            print(f'Index saved to {self.pickle_file}')

    def __len__(self) -> int:
        return len(self.index)

    def find_similar_names(self, query: str) -> Iterator[str]:
        """
        Find character names containing the query string (case-insensitive).
        """
        query = query.lower()
        for char, name in self.index.items():
            if query in name.lower():
                yield char

    def find_description_strs(self, query: str) -> Iterator[str]:
        """
        Generate description strings for characters matching the query.
        """
        for char in self.find_similar_names(query):
            code = ord(char)
            name = unicodedata.name(char, '')
            yield f'U+{code:04X}\t{char}\t{name}'

    def status(self, query: str, found: int) -> str:
        """
        Generate status message for the query results.
        """
        if found:
            return f'{found} matches for {query!r}'
        else:
            return f'no matches for {query!r}'


async def handle_queries(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter) -> None:
    """
    Coroutine to handle a single client connection.

    Implements the query-response protocol:
    1. Send prompt
    2. Read query
    3. Send results
    4. Repeat until client disconnects
    """
    while True:
        writer.write(PROMPT)  # Send prompt
        await writer.drain()  # Wait for prompt to be sent

        data = await reader.readline()  # Read query line
        if not data:  # Empty line = client disconnected
            break

        try:
            query = data.decode().strip()
        except UnicodeDecodeError:
            query = '\x00'  # Invalid UTF-8 becomes null char

        client = writer.get_extra_info('peername')
        print(f'Received from {client}: {query!r}')

        # Control characters signal end of session
        if ord(query[:1]) < 32:
            break

        # Search and send results
        lines = list(index.find_description_strs(query))
        if lines:
            writer.writelines(line.encode() + CRLF for line in lines)

        # Send status line
        status_line = index.status(query, len(lines))
        writer.write(status_line.encode() + CRLF)
        await writer.drain()

        print(f'Sent {len(lines)} results')

    print('Close client socket')
    writer.close()


async def run_server(address: str = '127.0.0.1', port: int = 2323) -> None:
    """
    Async server function.

    Starts the asyncio TCP server and runs until interrupted.
    """
    # Create server
    server = await asyncio.start_server(handle_queries, address, port)

    # Get actual host/port
    host = server.sockets[0].getsockname()
    print(f'Serving on {host}. Hit CTRL-C to stop.')
    print(f'Index contains {len(index)} Unicode characters.')
    print('Example queries: "chess", "arrow", "greek", "smiley"')

    try:
        # Run server indefinitely
        await server.serve_forever()
    except KeyboardInterrupt:
        pass

    print('Server shutting down.')
    server.close()
    await server.wait_closed()


def main(address: str = '127.0.0.1', port: str = '2323') -> None:
    """
    Main server function.

    Starts the asyncio TCP server using modern asyncio API.
    """
    port = int(port)
    asyncio.run(run_server(address, port))


# Global index instance (built on first import)
index = UnicodeNameIndex()


if __name__ == '__main__':
    main(*sys.argv[1:])
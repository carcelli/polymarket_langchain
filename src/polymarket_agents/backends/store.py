"""
Store Backend for Polymarket Agents

Provides cross-session persistent storage using LangGraph Store.
Perfect for long-term memories and analysis patterns.
"""

import json
from typing import List, Optional, Dict, Any, TYPE_CHECKING
from datetime import datetime

from deepagents.backends.protocol import BackendProtocol, WriteResult, EditResult
from deepagents.backends.utils import FileInfo, GrepMatch

if TYPE_CHECKING:
    from langgraph.store.base import BaseStore


class PolymarketStoreBackend(BackendProtocol):
    """
    LangGraph Store backend for persistent agent memories.

    Features:
    - Cross-thread/session persistence
    - Automatic versioning
    - Metadata-rich storage
    - Optimized for agent memory patterns
    """

    def __init__(self, runtime, namespace: str = "polymarket_agent"):
        """
        Initialize store backend.

        Args:
            runtime: ToolRuntime with access to store
            namespace: Namespace for this agent's data
        """
        self.runtime = runtime
        self.namespace = namespace
        self.store: "BaseStore" = runtime.store

    def _get_key(self, path: str) -> str:
        """Convert virtual path to store key."""
        # Remove leading slash and use as key
        key = path.lstrip("/")
        return f"{self.namespace}:{key}"

    def _get_path(self, key: str) -> str:
        """Convert store key back to virtual path."""
        # Remove namespace prefix
        if key.startswith(f"{self.namespace}:"):
            path = key[len(f"{self.namespace}:") :]
            return f"/{path}"
        return f"/{key}"

    def _get_item(self, path: str) -> Optional[Dict[str, Any]]:
        """Get item from store."""
        try:
            key = self._get_key(path)
            items = self.store.search([self.namespace], key=key)
            return items[0] if items else None
        except:
            return None

    def ls_info(self, path: str) -> List[FileInfo]:
        """List files and directories in store."""
        try:
            # Search for items under this path
            prefix = path.strip("/")
            if prefix and not prefix.endswith("/"):
                prefix += "/"

            items = self.store.search([self.namespace])
            entries = []

            for item in items:
                item_path = self._get_path(item.key)

                # Check if item is under requested path
                if not item_path.startswith(path.rstrip("/") + "/"):
                    continue

                # Extract metadata
                metadata = item.value.get("metadata", {})

                entries.append(
                    FileInfo(
                        path=item_path,
                        is_dir=False,  # Store items are always files
                        size=len(item.value.get("content", "")),
                        modified_at=metadata.get(
                            "modified_at", datetime.now().isoformat()
                        ),
                    )
                )

            # Sort by path
            entries.sort(key=lambda x: x.path)
            return entries

        except Exception:
            return []

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read content from store."""
        try:
            item = self._get_item(file_path)
            if not item:
                return f"Error: File '{file_path}' not found"

            content = item.value.get("content", "")

            # Apply offset and limit
            if offset > 0:
                content = content[offset:]

            if limit > 0 and len(content) > limit:
                content = (
                    content[:limit]
                    + f"\n... ({len(content) - limit} characters truncated)"
                )

            return content

        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}"

    def grep_raw(
        self, pattern: str, path: str | None = None, glob: str | None = None
    ) -> List[GrepMatch] | str:
        """Search for pattern in stored content."""
        try:
            import re

            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        matches = []

        try:
            items = self.store.search([self.namespace])

            for item in items[:50]:  # Limit search
                item_path = self._get_path(item.key)

                # Check path filters
                if path and not item_path.startswith(path):
                    continue
                if glob and glob not in item_path:
                    continue

                content = item.value.get("content", "")
                lines = content.split("\n")

                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        matches.append(
                            GrepMatch(path=item_path, line=line_num, text=line.rstrip())
                        )

                        if len(matches) >= 100:  # Limit results
                            break

                if len(matches) >= 100:
                    break

        except Exception:
            pass

        return matches

    def glob_info(self, pattern: str, path: str = "/") -> List[FileInfo]:
        """Find files matching glob pattern."""
        import fnmatch

        try:
            items = self.store.search([self.namespace])
            matches = []

            for item in items:
                item_path = self._get_path(item.key)

                # Apply glob pattern
                rel_path = item_path[len(path) :].lstrip("/")
                if fnmatch.fnmatch(rel_path, pattern):
                    metadata = item.value.get("metadata", {})
                    matches.append(
                        FileInfo(
                            path=item_path,
                            is_dir=False,
                            size=len(item.value.get("content", "")),
                            modified_at=metadata.get(
                                "modified_at", datetime.now().isoformat()
                            ),
                        )
                    )

            return matches[:50]

        except Exception:
            return []

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write new file to store (create-only)."""
        try:
            key = self._get_key(file_path)

            # Check if already exists
            existing = self.store.search([self.namespace], key=key)
            if existing:
                return WriteResult(error=f"File '{file_path}' already exists")

            # Store the item
            item_data = {
                "content": content,
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "modified_at": datetime.now().isoformat(),
                    "size": len(content),
                    "type": "file",
                },
            }

            self.store.put([self.namespace], key, item_data)

            return WriteResult(path=file_path, files_update=None)

        except Exception as e:
            return WriteResult(error=f"Failed to write file '{file_path}': {str(e)}")

    def edit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        """Edit existing file in store."""
        try:
            item = self._get_item(file_path)
            if not item:
                return EditResult(error=f"File '{file_path}' not found")

            content = item.value.get("content", "")

            # Find and replace
            if replace_all:
                new_content = content.replace(old_string, new_string)
                occurrences = content.count(old_string)
            else:
                index = content.find(old_string)
                if index == -1:
                    return EditResult(
                        error=f"String '{old_string}' not found in '{file_path}'"
                    )

                new_content = (
                    content[:index] + new_string + content[index + len(old_string) :]
                )
                occurrences = 1

            # Update in store
            item.value["content"] = new_content
            item.value["metadata"]["modified_at"] = datetime.now().isoformat()
            item.value["metadata"]["size"] = len(new_content)

            key = self._get_key(file_path)
            self.store.put([self.namespace], key, item.value)

            return EditResult(
                path=file_path, files_update=None, occurrences=occurrences
            )

        except Exception as e:
            return EditResult(error=f"Failed to edit file '{file_path}': {str(e)}")

    def store_memory(
        self,
        memory_type: str,
        content: str,
        tags: List[str] = None,
        context: Dict[str, Any] = None,
    ) -> Optional[str]:
        """Store a structured memory with rich metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"memories/{memory_type}_{timestamp}.json"
        filepath = f"/{filename}"

        memory_data = {
            "type": memory_type,
            "content": content,
            "tags": tags or [],
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
            "version": 1,
        }

        result = self.write(filepath, json.dumps(memory_data, indent=2, default=str))
        return result.path if not result.error else None

    def store_analysis_pattern(
        self,
        pattern_name: str,
        pattern_data: Dict[str, Any],
        confidence: float,
        examples: List[str],
    ) -> Optional[str]:
        """Store a learned analysis pattern."""
        filename = (
            f"patterns/{pattern_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        filepath = f"/{filename}"

        pattern = {
            "name": pattern_name,
            "data": pattern_data,
            "confidence": confidence,
            "examples": examples,
            "created_at": datetime.now().isoformat(),
            "usage_count": 0,
        }

        result = self.write(filepath, json.dumps(pattern, indent=2, default=str))
        return result.path if not result.error else None

    def get_memories_by_type(
        self, memory_type: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Retrieve memories of a specific type."""
        try:
            items = self.store.search([self.namespace])
            memories = []

            for item in items:
                if f"memories/{memory_type}_" in item.key:
                    try:
                        memory_data = json.loads(item.value.get("content", "{}"))
                        memories.append(memory_data)
                    except:
                        continue

            # Sort by timestamp, most recent first
            memories.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return memories[:limit]

        except Exception:
            return []

    def search_memories(
        self, query: str, tags: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Search memories by content and tags."""
        try:
            items = self.store.search([self.namespace])
            matches = []

            for item in items:
                if "memories/" not in item.key:
                    continue

                try:
                    memory_data = json.loads(item.value.get("content", "{}"))

                    # Check content match
                    content_match = (
                        query.lower() in memory_data.get("content", "").lower()
                    )

                    # Check tag match
                    tag_match = True
                    if tags:
                        memory_tags = set(memory_data.get("tags", []))
                        query_tags = set(tags)
                        tag_match = bool(memory_tags.intersection(query_tags))

                    if content_match and tag_match:
                        matches.append(memory_data)

                except:
                    continue

            return matches[:20]  # Limit results

        except Exception:
            return []

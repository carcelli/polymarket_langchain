"""
Filesystem Backend for Polymarket Agents

Provides persistent local storage for agent memories and analysis results.
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from deepagents.backends.protocol import BackendProtocol, WriteResult, EditResult
from deepagents.backends.utils import FileInfo, GrepMatch


class PolymarketFilesystemBackend(BackendProtocol):
    """
    Filesystem backend specifically designed for Polymarket trading agents.

    Features:
    - Automatic directory structure for different data types
    - Metadata tracking for analysis results
    - Compression for large datasets
    - Backup and versioning support
    """

    def __init__(self, root_dir: str, virtual_mode: bool = True):
        """
        Initialize filesystem backend.

        Args:
            root_dir: Root directory for storage
            virtual_mode: Whether to use virtual path mapping
        """
        self.root_dir = Path(root_dir).resolve()
        self.virtual_mode = virtual_mode

        # Create directory structure
        self._create_directories()

        # Track file metadata
        self.metadata_file = self.root_dir / ".metadata.json"
        self._load_metadata()

    def _create_directories(self):
        """Create the standard directory structure."""
        dirs = [
            "memories",      # Agent memories and learnings
            "analyses",      # Market analysis results
            "strategies",    # Trading strategies
            "market_data",   # Cached market data
            "logs",         # Agent operation logs
            "backups"       # Automatic backups
        ]

        for dir_name in dirs:
            (self.root_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def _load_metadata(self):
        """Load file metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self):
        """Save file metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)

    def _real_path(self, virtual_path: str) -> Path:
        """Convert virtual path to real filesystem path."""
        if not virtual_path.startswith('/'):
            virtual_path = '/' + virtual_path

        # Remove leading slash and resolve
        rel_path = virtual_path.lstrip('/')
        real_path = (self.root_dir / rel_path).resolve()

        # Security: ensure path is within root_dir
        try:
            real_path.relative_to(self.root_dir)
        except ValueError:
            raise ValueError(f"Access denied: {virtual_path} is outside allowed directory")

        return real_path

    def _virtual_path(self, real_path: Path) -> str:
        """Convert real path to virtual path."""
        try:
            rel_path = real_path.relative_to(self.root_dir)
            return '/' + str(rel_path)
        except ValueError:
            return str(real_path)

    def ls_info(self, path: str) -> List[FileInfo]:
        """List files and directories."""
        try:
            real_path = self._real_path(path)
            if not real_path.exists():
                return []

            entries = []
            for item in real_path.iterdir():
                if item.name.startswith('.'):  # Skip hidden files
                    continue

                stat = item.stat()
                entries.append(FileInfo(
                    path=self._virtual_path(item),
                    is_dir=item.is_dir(),
                    size=stat.st_size if item.is_file() else 0,
                    modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat()
                ))

            # Sort by path for consistent output
            entries.sort(key=lambda x: x.path)
            return entries

        except Exception as e:
            return []

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read file content with optional offset and limit."""
        try:
            real_path = self._real_path(file_path)

            if not real_path.exists() or not real_path.is_file():
                return f"Error: File '{file_path}' not found"

            with open(real_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Apply offset and limit
            if offset > 0:
                content = content[offset:]

            if limit > 0 and len(content) > limit:
                content = content[:limit] + f"\n... ({len(content) - limit} characters truncated)"

            return content

        except Exception as e:
            return f"Error reading file '{file_path}': {str(e)}"

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> List[GrepMatch] | str:
        """Search for pattern in files."""
        try:
            import re
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        matches = []

        # Determine which files to search
        if path:
            search_paths = [self._real_path(path)]
        elif glob:
            # Simple glob implementation
            search_paths = []
            base_path = self._real_path('/')
            for item in base_path.rglob('*'):
                if item.is_file() and (not glob or glob in str(item.relative_to(base_path))):
                    search_paths.append(item)
        else:
            # Search all files
            search_paths = []
            for item in self._real_path('/').rglob('*'):
                if item.is_file():
                    search_paths.append(item)

        # Search each file
        for file_path in search_paths[:50]:  # Limit to prevent excessive searching
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()

                for line_num, line in enumerate(lines, 1):
                    if regex.search(line):
                        matches.append(GrepMatch(
                            path=self._virtual_path(file_path),
                            line=line_num,
                            text=line.rstrip()
                        ))

                        # Limit matches to prevent overwhelming output
                        if len(matches) >= 100:
                            break

            except:
                continue

            if len(matches) >= 100:
                break

        return matches

    def glob_info(self, pattern: str, path: str = "/") -> List[FileInfo]:
        """Find files matching glob pattern."""
        import fnmatch

        try:
            base_path = self._real_path(path)
            matches = []

            # Simple glob implementation
            for item in base_path.rglob('*'):
                rel_path = str(item.relative_to(base_path))
                if fnmatch.fnmatch(rel_path, pattern):
                    stat = item.stat()
                    matches.append(FileInfo(
                        path=self._virtual_path(item),
                        is_dir=item.is_dir(),
                        size=stat.st_size if item.is_file() else 0,
                        modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat()
                    ))

            return matches[:50]  # Limit results

        except Exception:
            return []

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write new file (create-only)."""
        try:
            real_path = self._real_path(file_path)

            # Check if file already exists
            if real_path.exists():
                return WriteResult(error=f"File '{file_path}' already exists")

            # Ensure parent directory exists
            real_path.parent.mkdir(parents=True, exist_ok=True)

            # Write file
            with open(real_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Update metadata
            self.metadata[file_path] = {
                'created_at': datetime.now().isoformat(),
                'size': len(content),
                'type': 'file'
            }
            self._save_metadata()

            return WriteResult(path=file_path, files_update=None)

        except Exception as e:
            return WriteResult(error=f"Failed to write file '{file_path}': {str(e)}")

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """Edit existing file."""
        try:
            real_path = self._real_path(file_path)

            if not real_path.exists() or not real_path.is_file():
                return EditResult(error=f"File '{file_path}' not found")

            # Read current content
            with open(real_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find and replace
            if replace_all:
                new_content = content.replace(old_string, new_string)
                occurrences = content.count(old_string)
            else:
                # Replace only first occurrence
                index = content.find(old_string)
                if index == -1:
                    return EditResult(error=f"String '{old_string}' not found in '{file_path}'")

                new_content = content[:index] + new_string + content[index + len(old_string):]
                occurrences = 1

            # Write back
            with open(real_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            # Update metadata
            self.metadata[file_path]['modified_at'] = datetime.now().isoformat()
            self.metadata[file_path]['size'] = len(new_content)
            self._save_metadata()

            return EditResult(path=file_path, files_update=None, occurrences=occurrences)

        except Exception as e:
            return EditResult(error=f"Failed to edit file '{file_path}': {str(e)}")

    def store_analysis_result(self, market_question: str, analysis: Dict[str, Any]) -> str:
        """Store a market analysis result with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create filename from market question
        safe_name = "".join(c for c in market_question[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_name = safe_name.replace(' ', '_')

        filename = f"analyses/analysis_{safe_name}_{timestamp}.json"
        filepath = f"/{filename}"

        # Add metadata
        analysis_data = {
            'market_question': market_question,
            'timestamp': datetime.now().isoformat(),
            'analysis': analysis
        }

        # Write to backend
        result = self.write(filepath, json.dumps(analysis_data, indent=2, default=str))
        return result.path if not result.error else None

    def store_memory(self, memory_type: str, content: str, tags: List[str] = None) -> str:
        """Store an agent memory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"memories/{memory_type}_{timestamp}.md"
        filepath = f"/{filename}"

        # Format as markdown
        memory_content = f"# {memory_type.title()} Memory\n\n"
        memory_content += f"**Timestamp:** {datetime.now().isoformat()}\n\n"
        if tags:
            memory_content += f"**Tags:** {', '.join(tags)}\n\n"
        memory_content += f"## Content\n\n{content}\n"

        result = self.write(filepath, memory_content)
        return result.path if not result.error else None

    def get_recent_analyses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis results."""
        analyses_dir = self.root_dir / "analyses"
        if not analyses_dir.exists():
            return []

        analysis_files = list(analyses_dir.glob("*.json"))
        analysis_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        results = []
        for file_path in analysis_files[:limit]:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                results.append(data)
            except:
                continue

        return results

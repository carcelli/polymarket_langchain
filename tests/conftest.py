import os
import sys
from pathlib import Path


def pytest_sessionstart(session) -> None:
    if os.environ.get("PYTEST_ALLOW_LANGSMITH") == "1":
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

    # Include locally-installed packages (e.g. typer, rich) that the system Python
    # doesn't provide in an externally-managed environment (PEP 668).
    local_pkg_dir = Path(__file__).parent.parent / ".pip_packages"
    if local_pkg_dir.exists():
        pip_path = str(local_pkg_dir)
        if pip_path not in sys.path:
            # Keep editable-env/site-packages precedence; use this only as fallback.
            sys.path.append(pip_path)

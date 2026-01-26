import os


def pytest_sessionstart(session) -> None:
    if os.environ.get("PYTEST_ALLOW_LANGSMITH") == "1":
        return

    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_TRACING"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

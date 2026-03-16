"""
Async Utilities - Robust Coroutine Exception Handling (Fluent Python Chapter 16)

Provides patterns for building production-ready asyncio agents with proper:
- Exception handling (recoverable vs fatal errors)
- Resource cleanup (try/finally patterns)
- Task cancellation handling
- Restart and retry logic

Key Concepts from Chapter 16:
- Coroutines as cooperative state machines
- Exception propagation through yield points
- Cleanup with try/finally (not just except)
- Cancellation via CancelledError at yield points

Usage Patterns:
    # Basic task wrapper
    task = asyncio.create_task(
        robust_task("market_monitor", monitor_market(market_id))
    )

    # Custom exception handling
    async def my_task():
        try:
            while True:
                await do_work()
        except asyncio.CancelledError:
            await cleanup()
            raise  # Must re-raise
        except RecoverableError:
            logger.warning("Recovering from error...")
            await asyncio.sleep(5)
        finally:
            await mandatory_cleanup()

    # Agent supervisor pattern
    tasks = [
        asyncio.create_task(robust_task(f"monitor_{market_id}", monitor_market(market_id)))
        for market_id in market_ids
    ]
    await asyncio.gather(*tasks, return_exceptions=True)
"""

import asyncio
import inspect
import logging
from types import FrameType
from typing import Awaitable, Optional, Callable, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class _CoroutineReuseError(RuntimeError):
    """Raised when retry logic cannot recreate a one-shot coroutine."""


def _extract_call_args(
    func: Callable[..., Awaitable[Any]], frame_locals: dict[str, Any]
) -> tuple[list[Any], dict[str, Any]]:
    """Extract positional/keyword args for a function call from coroutine frame locals."""
    signature = inspect.signature(func)
    positional: list[Any] = []
    keywords: dict[str, Any] = {}

    for param in signature.parameters.values():
        name = param.name

        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            if name in frame_locals:
                positional.append(frame_locals[name])
            elif param.default is inspect.Signature.empty:
                raise ValueError(f"Missing positional-only argument: {name}")

        elif param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            if name in frame_locals:
                positional.append(frame_locals[name])
            elif param.default is inspect.Signature.empty:
                raise ValueError(f"Missing positional argument: {name}")

        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            if name in frame_locals:
                positional.extend(frame_locals[name])

        elif param.kind is inspect.Parameter.KEYWORD_ONLY:
            if name in frame_locals:
                keywords[name] = frame_locals[name]
            elif param.default is inspect.Signature.empty:
                raise ValueError(f"Missing keyword-only argument: {name}")

        elif param.kind is inspect.Parameter.VAR_KEYWORD:
            if name in frame_locals:
                keywords.update(frame_locals[name])

    return positional, keywords


def _build_retry_factory(
    task: Awaitable[Any], caller_frame: Optional[FrameType]
) -> Optional[Callable[[], Awaitable[Any]]]:
    """Build a coroutine factory for retries using caller-scope function resolution."""
    if caller_frame is None or not inspect.iscoroutine(task):
        return None

    code = task.cr_code
    frame_locals = dict(task.cr_frame.f_locals) if task.cr_frame is not None else {}

    for scope in (caller_frame.f_locals, caller_frame.f_globals):
        for value in scope.values():
            candidate = value.__func__ if inspect.ismethod(value) else value
            if inspect.iscoroutinefunction(candidate) and getattr(
                candidate, "__code__", None
            ) is code:
                try:
                    args, kwargs = _extract_call_args(candidate, frame_locals)
                except ValueError:
                    continue
                return lambda c=candidate, a=tuple(args), k=dict(kwargs): c(*a, **k)

    return None


def _resolve_coro_factory(
    task: Awaitable[Any] | Callable[[], Awaitable[Any]],
    caller_frame: Optional[FrameType],
) -> Callable[[], Awaitable[Any]]:
    """
    Normalize input into a coroutine factory for safe retries.

    If a coroutine object is passed, try to recover its originating coroutine
    function from the caller scope. If that is not possible, allow one run and
    raise a clear error on retry.
    """
    if callable(task) and not inspect.iscoroutine(task):
        return task

    if inspect.iscoroutine(task):
        retry_factory = _build_retry_factory(task, caller_frame)
        used_initial = False

        def coro_factory() -> Awaitable[Any]:
            nonlocal used_initial
            if not used_initial:
                used_initial = True
                return task
            if retry_factory is not None:
                return retry_factory()
            raise _CoroutineReuseError(
                "Cannot retry a one-shot coroutine object. Pass a coroutine "
                "factory (callable returning a new coroutine) to enable retries."
            )

        def close_pending() -> None:
            nonlocal used_initial
            if not used_initial:
                task.close()
                used_initial = True

        setattr(coro_factory, "close_pending", close_pending)
        return coro_factory

    raise TypeError(
        "task must be an awaitable or a callable returning an awaitable"
    )


def _cleanup_pending_coroutine(coro_factory: Callable[[], Awaitable[Any]]) -> None:
    """Close an unstarted coroutine to avoid RuntimeWarning on cancellation."""
    closer = getattr(coro_factory, "close_pending", None)
    if callable(closer):
        closer()


@dataclass
class TaskMetrics:
    """Metrics for monitoring task health and performance."""

    name: str
    start_time: datetime = field(default_factory=datetime.now)
    restarts: int = 0
    errors: int = 0
    last_error: Optional[Exception] = None
    last_error_time: Optional[datetime] = None
    total_runtime: timedelta = field(default_factory=lambda: timedelta(0))

    def record_error(self, exc: Exception) -> None:
        """Record an error occurrence."""
        self.errors += 1
        self.last_error = exc
        self.last_error_time = datetime.now()

    def record_restart(self) -> None:
        """Record a task restart."""
        self.restarts += 1

    def update_runtime(self) -> None:
        """Update total runtime tracking."""
        self.total_runtime = datetime.now() - self.start_time


class TaskSupervisor:
    """
    Supervisor for managing multiple async tasks with health monitoring.

    Provides centralized task lifecycle management, metrics collection,
    and graceful shutdown handling.
    """

    def __init__(self) -> None:
        self.tasks: dict[str, asyncio.Task] = {}
        self._task_factories: dict[str, Callable[[], Awaitable[Any]]] = {}
        self.metrics: dict[str, TaskMetrics] = {}
        self._shutdown_event = asyncio.Event()

    async def start_task(
        self,
        name: str,
        coro: Awaitable[Any] | Callable[[], Awaitable[Any]],
        restart_on_failure: bool = True,
    ) -> None:
        """Start a supervised task."""
        if name in self.tasks:
            logger.warning(f"Task {name} already exists, replacing...")

        caller_frame = inspect.currentframe().f_back
        try:
            coro_factory = _resolve_coro_factory(coro, caller_frame)
        finally:
            del caller_frame

        self.metrics[name] = TaskMetrics(name)
        self._task_factories[name] = coro_factory
        task = asyncio.create_task(
            self._run_supervised_task(name, coro_factory, restart_on_failure)
        )
        self.tasks[name] = task

        logger.info(f"Started supervised task: {name}")

    async def stop_task(self, name: str) -> None:
        """Stop a supervised task gracefully."""
        if name not in self.tasks:
            logger.warning(f"Task {name} not found")
            return

        task = self.tasks[name]
        if not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.CancelledError:
                pass
            except asyncio.TimeoutError:
                logger.warning(f"Task {name} did not stop gracefully")

        coro_factory = self._task_factories.pop(name, None)
        if coro_factory is not None:
            _cleanup_pending_coroutine(coro_factory)

        del self.tasks[name]
        logger.info(f"Stopped task: {name}")

    async def stop_all(self) -> None:
        """Stop all supervised tasks."""
        logger.info("Stopping all supervised tasks...")
        stop_tasks = [self.stop_task(name) for name in list(self.tasks.keys())]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        logger.info("All tasks stopped")

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()

    def signal_shutdown(self) -> None:
        """Signal all tasks to shut down."""
        self._shutdown_event.set()

    def get_metrics(self) -> dict[str, dict]:
        """Get current metrics for all tasks."""
        return {
            name: {
                "restarts": metrics.restarts,
                "errors": metrics.errors,
                "last_error": str(metrics.last_error) if metrics.last_error else None,
                "last_error_time": (
                    metrics.last_error_time.isoformat()
                    if metrics.last_error_time
                    else None
                ),
                "total_runtime_seconds": metrics.total_runtime.total_seconds(),
                "status": (
                    "running"
                    if name in self.tasks and not self.tasks[name].done()
                    else "stopped"
                ),
            }
            for name, metrics in self.metrics.items()
        }

    async def _run_supervised_task(
        self,
        name: str,
        coro_factory: Callable[[], Awaitable[Any]],
        restart_on_failure: bool,
    ) -> None:
        """Internal method to run a supervised task with error handling."""
        while not self._shutdown_event.is_set():
            try:
                logger.info(f"Starting supervised task iteration: {name}")
                await coro_factory()

                # Normal completion - don't restart
                logger.info(f"Task completed normally: {name}")
                break

            except asyncio.CancelledError:
                _cleanup_pending_coroutine(coro_factory)
                logger.info(f"Task cancelled: {name}")
                raise  # Allow cancellation to propagate

            except Exception as exc:
                self.metrics[name].record_error(exc)
                logger.error(f"Task failed: {name} | {exc!r}", exc_info=True)

                if not restart_on_failure or isinstance(exc, _CoroutineReuseError):
                    logger.error(f"Fatal error in task {name}, not restarting")
                    break

                # Exponential backoff for retries
                backoff_seconds = min(
                    2 ** self.metrics[name].errors, 300
                )  # Max 5 minutes
                logger.info(f"Restarting task {name} in {backoff_seconds}s...")
                self.metrics[name].record_restart()

                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=backoff_seconds
                    )
                    # Shutdown was requested during backoff
                    break
                except asyncio.TimeoutError:
                    # Backoff completed, restart the task
                    continue

        # Update final runtime
        if name in self.metrics:
            self.metrics[name].update_runtime()
        coro_factory_ref = self._task_factories.pop(name, None)
        if coro_factory_ref is not None:
            _cleanup_pending_coroutine(coro_factory_ref)


async def robust_task(
    name: str,
    coro: Awaitable[Any] | Callable[[], Awaitable[Any]],
    restart_on_failure: bool = True,
    backoff_seconds: float = 5.0,
) -> None:
    """
    Robust wrapper for agent tasks with proper exception handling and cleanup.

    This is the core pattern from Fluent Python Chapter 16 for coroutine exception handling:
    - Handles recoverable exceptions by retrying
    - Performs mandatory cleanup in finally blocks
    - Properly handles cancellation
    - Logs errors without crashing the supervisor

    Args:
        name: Descriptive name for the task (for logging)
        coro: The coroutine to run
        restart_on_failure: Whether to restart on recoverable exceptions
        backoff_seconds: Base delay between retries
    """
    caller_frame = inspect.currentframe().f_back
    try:
        coro_factory = _resolve_coro_factory(coro, caller_frame)
    finally:
        del caller_frame

    iteration = 0
    while True:
        iteration += 1
        try:
            logger.info(f"Starting task: {name} (iteration {iteration})")
            await coro_factory()
            logger.info(f"Task completed successfully: {name}")
            break  # Normal completion

        except asyncio.CancelledError:
            logger.warning(f"Task cancelled: {name}")
            raise  # Must re-raise to allow proper cancellation

        except Exception as exc:
            logger.error(f"Task failed: {name} | {exc!r}", exc_info=True)

            if isinstance(exc, _CoroutineReuseError):
                raise RuntimeError(str(exc)) from exc

            if not restart_on_failure:
                logger.error(f"Fatal error in task {name}, terminating")
                raise

            # Exponential backoff with jitter
            delay = min(backoff_seconds * (2 ** (iteration - 1)), 300)  # Max 5 minutes
            logger.info(f"Retrying task {name} in {delay:.1f}s")

            # Check for shutdown signal during backoff
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                logger.warning(f"Task cancelled during backoff: {name}")
                raise

        finally:
            logger.debug(f"Cleanup completed for task: {name}")


@asynccontextmanager
async def managed_resource(
    initialize: Callable[[], Awaitable[Any]], cleanup: Callable[[Any], Awaitable[None]]
):
    """
    Async context manager for resource lifecycle management.

    Ensures proper cleanup even when tasks are cancelled.

    Usage:
        async with managed_resource(init_websocket, close_websocket) as ws:
            async for message in ws:
                process(message)
    """
    resource = None
    try:
        resource = await initialize()
        yield resource
    finally:
        if resource is not None:
            try:
                await cleanup(resource)
            except Exception as exc:
                logger.error(f"Error during resource cleanup: {exc!r}")


# ===== EXAMPLE PATTERNS FOR AGENT COROUTINES =====


async def example_market_monitor(market_id: str) -> None:
    """
    Example of a properly structured agent coroutine with exception handling.

    Demonstrates the patterns from Fluent Python Chapter 16:
    - try/finally for mandatory cleanup
    - Specific exception handling for recoverable errors
    - Proper cancellation handling
    """
    websocket = None

    try:
        while True:
            try:
                # Resource acquisition (could fail)
                websocket = await connect_to_market_feed(market_id)
                logger.info(f"Connected to market feed: {market_id}")

                # Main processing loop
                async for message in websocket:
                    try:
                        analysis = await analyze_market_message(message)
                        if analysis.get("edge", 0) > 0.05:  # 5% edge threshold
                            await submit_trade_signal(market_id, analysis)
                    except ValueError as exc:
                        logger.warning(f"Invalid message format: {exc}")
                        continue  # Skip this message, continue processing

            except (ConnectionError, asyncio.TimeoutError) as exc:
                # Recoverable network errors - log and retry
                logger.warning(f"Network error for {market_id}: {exc}, retrying...")
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                # Task cancellation - cleanup and re-raise
                logger.info(f"Market monitor cancelled: {market_id}")
                break

            except Exception as exc:
                # Unexpected errors - log but don't crash
                logger.error(f"Unexpected error in market monitor {market_id}: {exc!r}")
                await asyncio.sleep(30)  # Longer backoff for unexpected errors

    finally:
        # Mandatory cleanup - runs even on cancellation
        if websocket:
            try:
                await websocket.close()
                logger.info(f"Closed websocket for {market_id}")
            except Exception as exc:
                logger.error(f"Error closing websocket for {market_id}: {exc!r}")


async def example_data_ingestion_worker(queue: asyncio.Queue) -> None:
    """
    Example of a queue-based worker with proper exception handling.
    """
    db_connection = None

    try:
        db_connection = await establish_db_connection()

        while True:
            try:
                # Get work item with timeout
                item = await asyncio.wait_for(queue.get(), timeout=30.0)
                queue.task_done()

                # Process the item
                await process_data_item(db_connection, item)
                logger.debug(f"Processed item: {item.get('id', 'unknown')}")

            except asyncio.TimeoutError:
                # No work available - check if we should exit
                if queue.empty():
                    logger.info("Queue empty, worker exiting")
                    break
                continue

            except ValidationError as exc:
                logger.warning(f"Invalid data item: {exc}, skipping")
                continue

    except asyncio.CancelledError:
        logger.info("Data ingestion worker cancelled")
        raise

    finally:
        if db_connection:
            await db_connection.close()
            logger.info("Database connection closed")


# ===== UTILITY FUNCTIONS (stubs for the examples) =====


async def connect_to_market_feed(market_id: str):
    """Stub for websocket connection."""
    await asyncio.sleep(0.1)  # Simulate connection time
    return f"mock_websocket_{market_id}"


async def analyze_market_message(message: str) -> dict:
    """Stub for message analysis."""
    return {"edge": 0.03, "signal": "hold"}


async def submit_trade_signal(market_id: str, analysis: dict) -> None:
    """Stub for trade submission."""
    logger.info(f"Submitted trade for {market_id}: {analysis}")


async def establish_db_connection():
    """Stub for database connection."""
    await asyncio.sleep(0.1)
    return "mock_db_connection"


async def process_data_item(connection, item: dict) -> None:
    """Stub for data processing."""
    await asyncio.sleep(0.01)


class ValidationError(Exception):
    """Custom validation error."""

    pass


# ===== DEMONSTRATION =====


async def demo_robust_tasks():
    """Demonstrate the robust task patterns."""
    print("🚀 Robust Task Exception Handling Demo")
    print("=" * 50)

    # Example 1: Basic robust task
    async def failing_task():
        for i in range(3):
            print(f"Task iteration {i}")
            if i == 1:
                raise ValueError("Simulated error")
            await asyncio.sleep(0.1)
        print("Task completed")

    print("\n1. Basic robust task with error recovery:")
    await robust_task("demo_task", failing_task(), restart_on_failure=False)

    # Example 2: Task supervisor
    print("\n2. Task supervisor with metrics:")
    supervisor = TaskSupervisor()

    async def monitored_task():
        await asyncio.sleep(0.5)
        raise ConnectionError("Network error")

    await supervisor.start_task("monitored", monitored_task())
    await asyncio.sleep(2)  # Let it run and restart

    metrics = supervisor.get_metrics()
    print(f"Task metrics: {metrics}")

    await supervisor.stop_all()

    print("\n✅ Exception handling patterns demonstrated!")


if __name__ == "__main__":
    asyncio.run(demo_robust_tasks())

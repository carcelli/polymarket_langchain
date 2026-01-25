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
import logging
from typing import Awaitable, Optional, Callable, Any
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


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

    def __init__(self):
        self.tasks: dict[str, asyncio.Task] = {}
        self.metrics: dict[str, TaskMetrics] = {}
        self._shutdown_event = asyncio.Event()

    async def start_task(
        self, name: str, coro: Awaitable, restart_on_failure: bool = True
    ) -> None:
        """Start a supervised task."""
        if name in self.tasks:
            logger.warning(f"Task {name} already exists, replacing...")

        self.metrics[name] = TaskMetrics(name)
        task = asyncio.create_task(
            self._run_supervised_task(name, coro, restart_on_failure)
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
            except asyncio.TimeoutError:
                logger.warning(f"Task {name} did not stop gracefully")

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
        self, name: str, coro: Awaitable, restart_on_failure: bool
    ) -> None:
        """Internal method to run a supervised task with error handling."""
        while not self._shutdown_event.is_set():
            try:
                logger.info(f"Starting supervised task iteration: {name}")
                await coro

                # Normal completion - don't restart
                logger.info(f"Task completed normally: {name}")
                break

            except asyncio.CancelledError:
                logger.info(f"Task cancelled: {name}")
                raise  # Allow cancellation to propagate

            except Exception as exc:
                self.metrics[name].record_error(exc)
                logger.error(f"Task failed: {name} | {exc!r}", exc_info=True)

                if not restart_on_failure:
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


async def robust_task(
    name: str,
    coro: Awaitable,
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
    iteration = 0

    while True:
        iteration += 1
        try:
            logger.info(f"Starting task: {name} (iteration {iteration})")
            await coro
            logger.info(f"Task completed successfully: {name}")
            break  # Normal completion

        except asyncio.CancelledError:
            logger.warning(f"Task cancelled: {name}")
            raise  # Must re-raise to allow proper cancellation

        except Exception as exc:
            logger.error(f"Task failed: {name} | {exc!r}", exc_info=True)

            if not restart_on_failure:
                logger.error(f"Fatal error in task {name}, terminating")
                break

            # Exponential backoff with jitter
            delay = min(backoff_seconds * (2 ** (iteration - 1)), 300)  # Max 5 minutes
            logger.info(".1f")

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

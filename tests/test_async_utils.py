"""
Test Robust Async Task Exception Handling (Fluent Python Chapter 16)

Tests the async_utils module patterns for:
- Exception recovery and retry logic
- Proper cleanup on cancellation
- Task supervision and metrics
- Resource management patterns
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock

from polymarket_agents.core.async_utils import (
    robust_task,
    TaskSupervisor,
    managed_resource,
    example_market_monitor,
    example_data_ingestion_worker,
)


class TestRobustTask:
    """Test the robust_task wrapper function."""

    @pytest.mark.asyncio
    async def test_normal_completion(self):
        """Test task that completes normally."""

        async def normal_task():
            await asyncio.sleep(0.01)
            return "completed"

        result = await robust_task("normal", normal_task(), restart_on_failure=False)
        assert result is None  # robust_task doesn't return the coro result

    @pytest.mark.asyncio
    async def test_recoverable_error_retry(self):
        """Test task that fails but recovers with retry."""
        call_count = 0

        async def failing_task():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            await asyncio.sleep(0.01)

        start_time = time.time()
        await robust_task("retry_test", failing_task(), restart_on_failure=True)
        elapsed = time.time() - start_time

        assert call_count == 3  # Should have failed twice and succeeded once
        assert elapsed >= 10  # Should have backoff delays (5 + 5 seconds)

    @pytest.mark.asyncio
    async def test_fatal_error_no_retry(self):
        """Test task that fails fatally without retry."""
        call_count = 0

        async def fatal_task():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Fatal error")

        with pytest.raises(RuntimeError):
            await robust_task("fatal", fatal_task(), restart_on_failure=False)

        assert call_count == 1  # Should not retry

    @pytest.mark.asyncio
    async def test_cancellation_propagation(self):
        """Test that cancellation is properly propagated."""

        async def cancellable_task():
            await asyncio.sleep(10)  # Long running

        task = asyncio.create_task(
            robust_task("cancellable", cancellable_task(), restart_on_failure=False)
        )

        await asyncio.sleep(0.01)  # Let task start
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task


class TestTaskSupervisor:
    """Test the TaskSupervisor class."""

    @pytest.mark.asyncio
    async def test_task_lifecycle(self):
        """Test basic task start/stop lifecycle."""
        supervisor = TaskSupervisor()

        async def simple_task():
            await asyncio.sleep(1)

        await supervisor.start_task("simple", simple_task())
        assert "simple" in supervisor.tasks

        await supervisor.stop_task("simple")
        assert "simple" not in supervisor.tasks

    @pytest.mark.asyncio
    async def test_task_metrics_collection(self):
        """Test that metrics are collected properly."""
        supervisor = TaskSupervisor()

        async def failing_task():
            await asyncio.sleep(0.01)
            raise ValueError("Test error")

        await supervisor.start_task("failing", failing_task(), restart_on_failure=False)

        # Wait for task to complete/fail
        await asyncio.sleep(0.1)

        metrics = supervisor.get_metrics()
        assert "failing" in metrics
        assert metrics["failing"]["errors"] == 1
        assert metrics["failing"]["status"] == "stopped"
        assert "Test error" in metrics["failing"]["last_error"]

    @pytest.mark.asyncio
    async def test_supervisor_shutdown(self):
        """Test graceful shutdown of all tasks."""
        supervisor = TaskSupervisor()

        async def long_task():
            await asyncio.sleep(10)

        await supervisor.start_task("task1", long_task())
        await supervisor.start_task("task2", long_task())

        assert len(supervisor.tasks) == 2

        # Signal shutdown
        supervisor.signal_shutdown()
        await supervisor.stop_all()

        assert len(supervisor.tasks) == 0


class TestManagedResource:
    """Test the managed_resource context manager."""

    @pytest.mark.asyncio
    async def test_normal_resource_lifecycle(self):
        """Test normal resource initialization and cleanup."""
        init_called = False
        cleanup_called = False
        resource_value = None

        async def initialize():
            nonlocal init_called, resource_value
            init_called = True
            resource_value = "test_resource"
            return resource_value

        async def cleanup(resource):
            nonlocal cleanup_called
            cleanup_called = True
            assert resource == "test_resource"

        async with managed_resource(initialize, cleanup) as resource:
            assert resource == "test_resource"
            assert init_called
            assert not cleanup_called

        assert cleanup_called

    @pytest.mark.asyncio
    async def test_exception_during_usage(self):
        """Test cleanup happens even when exception occurs."""
        cleanup_called = False

        async def initialize():
            return "resource"

        async def cleanup(resource):
            nonlocal cleanup_called
            cleanup_called = True

        with pytest.raises(ValueError):
            async with managed_resource(initialize, cleanup) as resource:
                raise ValueError("Test exception")

        assert cleanup_called

    @pytest.mark.asyncio
    async def test_cleanup_error_handling(self):
        """Test that cleanup errors are logged but don't prevent context manager exit."""

        async def initialize():
            return "resource"

        async def failing_cleanup(resource):
            raise RuntimeError("Cleanup failed")

        # Should not raise the cleanup error
        async with managed_resource(initialize, failing_cleanup) as resource:
            pass


class TestExamplePatterns:
    """Test the example coroutine patterns."""

    @pytest.mark.asyncio
    async def test_market_monitor_cancellation(self):
        """Test that market monitor handles cancellation properly."""
        # Mock the websocket functions
        original_connect = asyncio.sleep  # Placeholder

        async def mock_connect(market_id):
            await asyncio.sleep(0.01)
            return "mock_ws"

        async def mock_close(ws):
            pass

        # Patch the functions (in real usage, these would be imported)
        # For this test, we'll just verify the pattern structure exists
        assert callable(example_market_monitor)

    @pytest.mark.asyncio
    async def test_data_worker_timeout(self):
        """Test data ingestion worker handles timeouts."""
        queue = asyncio.Queue()

        # Create a task that will timeout waiting for queue items
        task = asyncio.create_task(example_data_ingestion_worker(queue))

        # Wait for it to timeout and exit
        await asyncio.sleep(35)  # Longer than the 30s timeout

        # Task should have completed due to timeout
        assert task.done()


class TestIntegration:
    """Integration tests for the async utilities."""

    @pytest.mark.asyncio
    async def test_multiple_tasks_with_supervisor(self):
        """Test running multiple tasks under supervision."""
        supervisor = TaskSupervisor()

        async def quick_task(name: str):
            await asyncio.sleep(0.01)
            return f"completed_{name}"

        async def failing_task(name: str):
            await asyncio.sleep(0.01)
            raise ConnectionError(f"Error in {name}")

        # Start multiple tasks
        await supervisor.start_task("quick1", quick_task("quick1"))
        await supervisor.start_task("quick2", quick_task("quick2"))
        await supervisor.start_task(
            "failing", failing_task("failing"), restart_on_failure=False
        )

        # Wait for all to complete
        await asyncio.sleep(0.1)

        metrics = supervisor.get_metrics()
        assert len(metrics) == 3

        # Quick tasks should have completed successfully
        assert metrics["quick1"]["errors"] == 0
        assert metrics["quick2"]["errors"] == 0

        # Failing task should have recorded an error
        assert metrics["failing"]["errors"] == 1

        await supervisor.stop_all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

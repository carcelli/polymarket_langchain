# Coroutine Exception Handling Patterns (Fluent Python Chapter 16)

Production-ready patterns for building asyncio agents with robust exception handling, cleanup, and cancellation support.

## Overview

This module provides battle-tested patterns for asyncio coroutine exception handling based on Fluent Python Chapter 16. The key insight is that coroutines are cooperative state machines, not threads—exceptions behave differently and require careful handling.

## Key Concepts from Chapter 16

### 1. **Exception Propagation Through Yield Points**
```python
async def coro():
    await step1()  # Exception here propagates normally
    data = await step2()  # Exception here also propagates normally
    await step3(data)

# Unlike threads, exceptions in coroutines:
# - Are raised at the await/yield point
# - Can be caught by surrounding try/except
# - Terminate the coroutine if unhandled
```

### 2. **Cancellation via CancelledError**
```python
async def cancellable_coro():
    try:
        while True:
            await do_work()
    except asyncio.CancelledError:
        await cleanup()  # Mandatory cleanup
        raise  # Must re-raise to complete cancellation
```

### 3. **Cleanup with try/finally (Not except)**
```python
async def resource_user():
    resource = await acquire_resource()
    try:
        await use_resource(resource)
    finally:
        await release_resource(resource)  # Runs even on cancellation
```

## Core Components

### `robust_task()` - Task Wrapper

The foundation for all agent coroutines. Provides automatic retry, logging, and cleanup.

```python
from polymarket_agents.core.async_utils import robust_task

# Basic usage
await robust_task("market_monitor", monitor_market(market_id))

# With custom settings
await robust_task(
    "data_processor",
    process_data_stream(),
    restart_on_failure=True,  # Retry on errors
    backoff_seconds=10.0     # Custom backoff
)
```

**Features:**
- Automatic retry on recoverable exceptions
- Exponential backoff (5s, 10s, 20s, max 5min)
- Proper cancellation handling
- Comprehensive logging
- No crash on task failure

### `TaskSupervisor` - Multi-Task Management

Centralized supervision of multiple async tasks with health monitoring.

```python
from polymarket_agents.core.async_utils import TaskSupervisor

supervisor = TaskSupervisor()

# Start multiple tasks
await supervisor.start_task("monitor_btc", monitor_market("btc"))
await supervisor.start_task("monitor_eth", monitor_market("eth"))
await supervisor.start_task("data_ingest", ingest_market_data())

# Get health metrics
metrics = supervisor.get_metrics()
# {
#   "monitor_btc": {
#     "restarts": 2,
#     "errors": 1,
#     "last_error": "ConnectionError: Network timeout",
#     "status": "running"
#   }
# }

# Graceful shutdown
supervisor.signal_shutdown()
await supervisor.stop_all()
```

### `managed_resource()` - Resource Lifecycle

Async context manager ensuring proper resource cleanup.

```python
from polymarket_agents.core.async_utils import managed_resource

async def init_websocket(url):
    return await connect_websocket(url)

async def close_websocket(ws):
    await ws.close()

async with managed_resource(
    lambda: init_websocket("ws://market.feed"),
    close_websocket
) as websocket:
    async for message in websocket:
        await process_message(message)
# Cleanup happens automatically, even on cancellation
```

## Usage Patterns for Agents

### 1. Market Monitoring Task

```python
async def monitor_market(market_id: str):
    """Production-ready market monitoring with proper exception handling."""
    websocket = None

    try:
        while True:
            try:
                # Resource acquisition (can fail)
                websocket = await connect_market_feed(market_id)

                # Main processing loop
                async for message in websocket:
                    try:
                        analysis = await analyze_message(message)
                        if analysis["edge"] > 0.05:  # 5% edge threshold
                            await submit_trade(market_id, analysis)
                    except ValueError:
                        logger.warning("Invalid message format, skipping")
                        continue

            except (ConnectionError, asyncio.TimeoutError):
                # Recoverable - log and retry
                logger.warning(f"Network error for {market_id}, retrying...")
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                # Task cancelled - cleanup and exit
                logger.info(f"Monitor cancelled for {market_id}")
                break

            except Exception as exc:
                # Unexpected - log and backoff longer
                logger.error(f"Unexpected error monitoring {market_id}: {exc}")
                await asyncio.sleep(60)

    finally:
        # Mandatory cleanup - runs on normal exit, error, or cancellation
        if websocket:
            await websocket.close()
            logger.info(f"Closed websocket for {market_id}")
```

### 2. Data Ingestion Worker

```python
async def data_ingestion_worker(queue: asyncio.Queue):
    """Queue-based worker with timeout and error handling."""
    db_connection = None

    try:
        db_connection = await establish_db_connection()

        while True:
            try:
                # Get work with timeout
                item = await asyncio.wait_for(queue.get(), timeout=30.0)
                queue.task_done()

                # Process item
                await process_data_item(db_connection, item)

            except asyncio.TimeoutError:
                # No work available
                if queue.empty():
                    logger.info("Queue empty, worker exiting")
                    break
                continue

            except ValidationError:
                logger.warning("Invalid data item, skipping")
                continue

    except asyncio.CancelledError:
        logger.info("Data ingestion worker cancelled")
        raise

    finally:
        if db_connection:
            await db_connection.close()
```

### 3. Agent Supervisor Pattern

```python
async def run_market_agents(market_ids: List[str]):
    """Supervisor pattern for managing multiple market agents."""
    supervisor = TaskSupervisor()

    # Start one monitor per market
    for market_id in market_ids:
        await supervisor.start_task(
            f"monitor_{market_id}",
            monitor_market(market_id)
        )

    # Monitor health
    while True:
        await asyncio.sleep(60)  # Check every minute
        metrics = supervisor.get_metrics()

        # Log concerning patterns
        for name, data in metrics.items():
            if data["errors"] > 10:  # Too many errors
                logger.error(f"Task {name} has {data['errors']} errors")
            if data["restarts"] > 5:  # Restarting too often
                logger.warning(f"Task {name} restarted {data['restarts']} times")

        # Check for shutdown signal
        if supervisor._shutdown_event.is_set():
            break

    await supervisor.stop_all()
```

## Exception Handling Strategies

### **Recoverable vs Fatal Exceptions**

```python
# Recoverable (retry)
except (ConnectionError, asyncio.TimeoutError, httpx.ConnectError):
    logger.warning("Network issue, will retry")
    await asyncio.sleep(backoff_seconds)

# Fatal (don't retry)
except (ValueError, KeyError, ConfigurationError):
    logger.error("Configuration/programming error, not retrying")
    raise  # Let supervisor handle

# Unexpected (log and backoff longer)
except Exception as exc:
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    await asyncio.sleep(300)  # 5 minute backoff
```

### **Cancellation Handling**

```python
# Always catch and re-raise CancelledError
except asyncio.CancelledError:
    logger.info("Task cancelled, cleaning up...")
    await cleanup_resources()
    raise  # Essential for proper cancellation
```

### **Resource Cleanup**

```python
# Use try/finally, not try/except
resource = await acquire_resource()
try:
    await use_resource(resource)
finally:
    await release_resource(resource)  # Runs always
```

## Testing Patterns

### Unit Testing Exception Handling

```python
@pytest.mark.asyncio
async def test_network_error_recovery():
    """Test that network errors trigger retry."""
    error_count = 0

    async def failing_monitor():
        nonlocal error_count
        error_count += 1
        if error_count < 3:
            raise ConnectionError("Network down")
        await asyncio.sleep(0.01)  # Success on 3rd try

    await robust_task("test_monitor", failing_monitor())
    assert error_count == 3  # Failed twice, succeeded once

@pytest.mark.asyncio
async def test_cancellation_cleanup():
    """Test cleanup happens on cancellation."""
    cleanup_called = False

    async def cancellable_task():
        nonlocal cleanup_called
        try:
            await asyncio.sleep(10)
        finally:
            cleanup_called = True

    task = asyncio.create_task(
        robust_task("test", cancellable_task(), restart_on_failure=False)
    )

    await asyncio.sleep(0.01)  # Let it start
    task.cancel()
    await task

    assert cleanup_called  # Cleanup must happen
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_supervisor_lifecycle():
    """Test full supervisor lifecycle."""
    supervisor = TaskSupervisor()

    async def quick_success():
        await asyncio.sleep(0.01)

    async def quick_failure():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")

    # Start tasks
    await supervisor.start_task("success", quick_success())
    await supervisor.start_task("failure", quick_failure(), restart_on_failure=False)

    # Wait for completion
    await asyncio.sleep(0.1)

    metrics = supervisor.get_metrics()
    assert metrics["success"]["errors"] == 0
    assert metrics["failure"]["errors"] == 1

    await supervisor.stop_all()
```

## Production Deployment

### Docker Integration

```dockerfile
# Add to docker-compose.yml
services:
  market_agents:
    build: .
    command: python -m polymarket_agents.core.async_utils
    environment:
      - LOG_LEVEL=INFO
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import asyncio; asyncio.run(check_agents())"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Health Monitoring

```python
async def health_check():
    """Agent health monitoring."""
    supervisor = TaskSupervisor()
    # ... start tasks ...

    while True:
        await asyncio.sleep(60)
        metrics = supervisor.get_metrics()

        # Report to monitoring system
        for name, data in metrics.items():
            if data["status"] == "stopped" and data["errors"] > 0:
                alert_system(f"Task {name} failed permanently")
            elif data["restarts"] > 10:
                alert_system(f"Task {name} restarting frequently")

        # Auto-restart failed tasks
        for name, data in metrics.items():
            if (data["status"] == "stopped" and
                data["last_error_time"] and
                (datetime.now() - data["last_error_time"]) > timedelta(minutes=5)):
                logger.info(f"Auto-restarting task {name}")
                await supervisor.start_task(name, original_coroutines[name])
```

### Graceful Shutdown

```python
async def main():
    supervisor = TaskSupervisor()

    # Setup signal handlers
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down...")
        supervisor.signal_shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Start agents
    await start_all_agents(supervisor)

    # Wait for shutdown
    await supervisor.wait_for_shutdown()
    await supervisor.stop_all()

if __name__ == "__main__":
    asyncio.run(main())
```

## Performance Considerations

- **Memory**: Each task has its own stack (~2KB baseline + variables)
- **CPU**: Minimal overhead for task switching vs threads
- **Concurrency**: Thousands of concurrent tasks possible
- **Monitoring**: Keep metrics collection lightweight

## Troubleshooting

### Task Won't Cancel

```python
# Problem: Not re-raising CancelledError
except asyncio.CancelledError:
    await cleanup()
    # Missing: raise

# Fix: Always re-raise
except asyncio.CancelledError:
    await cleanup()
    raise
```

### Resource Leaks

```python
# Problem: Cleanup in except instead of finally
try:
    resource = await acquire()
    await use(resource)
except Exception:
    await release(resource)  # Only runs on exception

# Fix: Use finally
resource = await acquire()
try:
    await use(resource)
finally:
    await release(resource)  # Runs always
```

### Silent Failures

```python
# Problem: Overly broad exception handling
except Exception:
    pass  # Silent failure

# Fix: Log and handle appropriately
except Exception as exc:
    logger.error(f"Task failed: {exc}", exc_info=True)
    if recoverable(exc):
        await asyncio.sleep(5)
    else:
        raise
```

---

These patterns transform asyncio agents from fragile prototypes to production-ready services. The key insight: treat coroutines as state machines that yield control cooperatively, requiring explicit exception handling at every yield point.

Your market monitoring agents will now survive network outages, database hiccups, and unexpected data formats without crashing the entire system. 🚀⚡
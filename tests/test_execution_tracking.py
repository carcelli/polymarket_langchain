"""
Tests for agent execution tracking and dashboard data layer.

Verifies that:
1. Execution tracking tables exist and are writable
2. MemoryManager methods work correctly
3. Dashboard can query execution data without errors
"""
import os
import tempfile
from pathlib import Path
import json
from datetime import datetime

import pytest

from polymarket_agents.memory.manager import MemoryManager


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_markets.db"
        yield str(db_path)


@pytest.fixture
def memory_manager(temp_db):
    """Create a MemoryManager with a temporary database."""
    return MemoryManager(db_path=temp_db)


class TestExecutionTracking:
    """Test execution tracking functionality."""

    def test_database_initialization(self, memory_manager):
        """Test that database initializes with all required tables."""
        stats = memory_manager.get_database_stats()
        assert stats is not None
        assert "total_markets" in stats
        assert "database_size_mb" in stats

    def test_start_agent_execution(self, memory_manager):
        """Test starting an agent execution."""
        execution_id = memory_manager.start_agent_execution(
            agent_type="memory_agent",
            agent_name="test_agent",
            query="Test query"
        )
        assert execution_id is not None
        assert isinstance(execution_id, int)
        assert execution_id > 0

    def test_complete_agent_execution(self, memory_manager):
        """Test completing an agent execution."""
        # Start execution
        execution_id = memory_manager.start_agent_execution(
            agent_type="memory_agent",
            agent_name="test_agent",
            query="Test query"
        )

        # Complete execution
        result = {"analysis": "test analysis", "decision": "PASS"}
        memory_manager.complete_agent_execution(
            execution_id=execution_id,
            result=json.dumps(result),
            tokens_used=100
        )

        # Verify completion
        executions = memory_manager.get_recent_executions(limit=1)
        assert len(executions) == 1
        assert executions[0]["id"] == execution_id
        assert executions[0]["status"] == "completed"
        assert executions[0]["tokens_used"] == 100

    def test_fail_agent_execution(self, memory_manager):
        """Test failing an agent execution."""
        # Start execution
        execution_id = memory_manager.start_agent_execution(
            agent_type="planning_agent",
            agent_name="test_agent",
            query="Test query"
        )

        # Fail execution
        error_msg = "Test error message"
        memory_manager.fail_agent_execution(execution_id, error=error_msg)

        # Verify failure
        executions = memory_manager.get_recent_executions(limit=1)
        assert len(executions) == 1
        assert executions[0]["id"] == execution_id
        assert executions[0]["status"] == "failed"
        assert error_msg in executions[0]["error"]

    def test_get_recent_executions(self, memory_manager):
        """Test retrieving recent executions."""
        # Create multiple executions
        for i in range(5):
            execution_id = memory_manager.start_agent_execution(
                agent_type="memory_agent",
                agent_name=f"test_agent_{i}",
                query=f"Test query {i}"
            )
            memory_manager.complete_agent_execution(
                execution_id=execution_id,
                result=json.dumps({"test": i}),
                tokens_used=i * 10
            )

        # Get recent executions
        executions = memory_manager.get_recent_executions(limit=3)
        assert len(executions) == 3
        # Should be in reverse chronological order
        assert executions[0]["query"] == "Test query 4"
        assert executions[1]["query"] == "Test query 3"
        assert executions[2]["query"] == "Test query 2"

    def test_get_execution_metrics_empty(self, memory_manager):
        """Test metrics when no executions exist."""
        metrics = memory_manager.get_execution_metrics(time_period="24h")
        assert metrics is None

    def test_get_execution_metrics_with_data(self, memory_manager):
        """Test metrics calculation with execution data."""
        # Create successful execution
        execution_id_1 = memory_manager.start_agent_execution(
            agent_type="memory_agent",
            agent_name="test_agent",
            query="Test query 1"
        )
        memory_manager.complete_agent_execution(
            execution_id=execution_id_1,
            result=json.dumps({"test": 1}),
            tokens_used=100
        )

        # Create failed execution
        execution_id_2 = memory_manager.start_agent_execution(
            agent_type="planning_agent",
            agent_name="test_agent",
            query="Test query 2"
        )
        memory_manager.fail_agent_execution(execution_id_2, error="Test error")

        # Get metrics
        metrics = memory_manager.get_execution_metrics(time_period="24h")
        assert metrics is not None
        assert metrics["total_runs"] == 2
        assert metrics["successful_runs"] == 1
        assert metrics["failed_runs"] == 1
        assert metrics["avg_tokens_used"] >= 0

    def test_track_node_execution(self, memory_manager):
        """Test tracking individual node executions."""
        # Start agent execution
        execution_id = memory_manager.start_agent_execution(
            agent_type="memory_agent",
            agent_name="test_agent",
            query="Test query"
        )

        # Track a node execution
        node_id = memory_manager.track_node_execution(
            agent_execution_id=execution_id,
            node_name="memory_node",
            node_type="retriever",
            input_data=json.dumps({"query": "test"}),
            output_data=json.dumps({"results": 5}),
            duration_ms=150,
            status="completed"
        )

        assert node_id is not None
        assert isinstance(node_id, int)
        assert node_id > 0

    def test_execution_tracking_full_flow(self, memory_manager):
        """Test complete execution tracking flow."""
        # Start execution
        execution_id = memory_manager.start_agent_execution(
            agent_type="memory_agent",
            agent_name="memory_agent",
            query="Find crypto markets"
        )

        # Track multiple nodes
        nodes = ["memory", "enrichment", "reasoning", "decision"]
        for node in nodes:
            memory_manager.track_node_execution(
                agent_execution_id=execution_id,
                node_name=node,
                node_type="processor",
                input_data="{}",
                output_data="{}",
                duration_ms=100,
                status="completed"
            )

        # Complete execution
        memory_manager.complete_agent_execution(
            execution_id=execution_id,
            result=json.dumps({"markets_found": 10}),
            tokens_used=500
        )

        # Verify
        executions = memory_manager.get_recent_executions(limit=1)
        assert len(executions) == 1
        assert executions[0]["status"] == "completed"
        assert executions[0]["tokens_used"] == 500


class TestDashboardDataLayer:
    """Test dashboard-specific queries."""

    def test_database_stats_structure(self, memory_manager):
        """Test that database stats return expected structure."""
        stats = memory_manager.get_database_stats()
        assert isinstance(stats, dict)
        assert "total_markets" in stats
        assert "database_size_mb" in stats
        assert isinstance(stats["total_markets"], int)
        assert isinstance(stats["database_size_mb"], float)

    def test_recent_executions_structure(self, memory_manager):
        """Test recent executions return expected structure."""
        # Create a test execution
        execution_id = memory_manager.start_agent_execution(
            agent_type="memory_agent",
            agent_name="test_agent",
            query="Test query"
        )
        memory_manager.complete_agent_execution(
            execution_id=execution_id,
            result=json.dumps({"test": "result"}),
            tokens_used=100
        )

        # Get executions
        executions = memory_manager.get_recent_executions(limit=10)
        assert isinstance(executions, list)
        assert len(executions) > 0

        # Check structure
        exec_data = executions[0]
        assert "id" in exec_data
        assert "agent_type" in exec_data
        assert "agent_name" in exec_data
        assert "query" in exec_data
        assert "status" in exec_data
        assert "started_at" in exec_data

    def test_top_volume_markets(self, memory_manager):
        """Test that top volume markets query works."""
        # Query top markets (should work even with empty database)
        top_markets = memory_manager.list_top_volume_markets(limit=5)
        assert isinstance(top_markets, list)
        # Empty list is valid for a fresh database


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

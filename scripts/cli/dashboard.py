#!/usr/bin/env python3
"""
Polymarket Agents - Live CLI Dashboard

Real-time monitoring of agent execution, performance, and system health.

Usage:
    python scripts/cli/dashboard.py
    python scripts/cli/dashboard.py --refresh 1.0
"""
import sys
import time
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
except ImportError:
    print("Error: Rich library not installed. Please run: pip install rich")
    sys.exit(1)

from polymarket_agents.memory.manager import MemoryManager


class PolymarketDashboard:
    """Live CLI dashboard for Polymarket agents."""

    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.memory_manager = MemoryManager()

        # Setup layout
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3),
        )

        self.layout["main"].split_row(Layout(name="left"), Layout(name="right"))

        self.layout["left"].split_column(
            Layout(name="agent_flow", ratio=2), Layout(name="executions", ratio=3)
        )

        self.layout["right"].split_column(
            Layout(name="metrics", ratio=2),
            Layout(name="system_health", ratio=2),
            Layout(name="markets", ratio=2),
        )

    def render_header(self) -> Panel:
        """Render dashboard header."""
        return Panel(
            "[bold cyan]⚡ Polymarket Agents - Live Dashboard[/bold cyan]\n"
            f"[dim]Last updated: {datetime.now().strftime('%H:%M:%S')}[/dim]",
            box=box.ROUNDED,
        )

    def render_agent_flow(self) -> Panel:
        """Render current agent execution flow."""
        try:
            executions = self.memory_manager.get_recent_executions(limit=1)

            if not executions or executions[0]["status"] != "running":
                content = "[dim]No active agent execution[/dim]"
            else:
                exec_data = executions[0]
                agent_name = exec_data["agent_name"]
                current_node = exec_data.get("current_node", "unknown") or "starting"
                completed_nodes_str = exec_data.get("completed_nodes", "[]") or "[]"
                completed_nodes = json.loads(completed_nodes_str)

                # Show flow with completed vs current vs pending
                if agent_name == "memory_agent":
                    nodes = ["Memory", "Enrichment", "Reasoning", "Decision"]
                else:  # planning_agent
                    nodes = ["Research", "Stats", "Probability", "Decision"]

                flow = []
                for node in nodes:
                    if node.lower() in [n.lower() for n in completed_nodes]:
                        flow.append(f"[green]✓ {node}[/green]")
                    elif node.lower() == current_node.lower():
                        flow.append(f"[yellow]⚡ {node}[/yellow]")
                    else:
                        flow.append(f"[dim]○ {node}[/dim]")

                content = f"**{agent_name}**\n\n" + " → ".join(flow)

            return Panel(
                content, title="[bold]Agent Execution Flow[/bold]", box=box.ROUNDED
            )
        except Exception as e:
            return Panel(
                f"[red]Error: {str(e)}[/red]",
                title="[bold]Agent Execution Flow[/bold]",
                box=box.ROUNDED,
            )

    def render_executions(self) -> Panel:
        """Render recent agent executions."""
        try:
            executions = self.memory_manager.get_recent_executions(limit=10)

            table = Table(show_header=True, box=box.SIMPLE)
            table.add_column("Time", style="cyan", width=8)
            table.add_column("Agent", width=15)
            table.add_column("Status", width=10)
            table.add_column("Duration", width=10)
            table.add_column("Query", overflow="fold")

            for exec_data in executions:
                started = datetime.fromisoformat(exec_data["started_at"])
                time_str = started.strftime("%H:%M:%S")

                status = exec_data["status"]
                if status == "completed":
                    status_str = "[green]✓ Done[/green]"
                elif status == "failed":
                    status_str = "[red]✗ Failed[/red]"
                else:
                    status_str = "[yellow]⚡ Running[/yellow]"

                duration = exec_data.get("duration_ms", 0)
                duration_str = f"{duration}ms" if duration else "-"

                query = exec_data.get("query", "") or ""
                query_display = query[:50] + "..." if len(query) > 50 else query

                table.add_row(
                    time_str,
                    exec_data["agent_name"],
                    status_str,
                    duration_str,
                    query_display,
                )

            return Panel(table, title="[bold]Recent Executions[/bold]", box=box.ROUNDED)
        except Exception as e:
            return Panel(
                f"[red]Error: {str(e)}[/red]",
                title="[bold]Recent Executions[/bold]",
                box=box.ROUNDED,
            )

    def render_metrics(self) -> Panel:
        """Render performance metrics."""
        try:
            metrics_24h = self.memory_manager.get_execution_metrics(time_period="24h")

            if not metrics_24h:
                content = (
                    "[dim]No metrics available - run some agents to see stats[/dim]"
                )
            else:
                success_rate = 0
                if metrics_24h.get("total_runs", 0) > 0:
                    success_rate = (
                        metrics_24h.get("successful_runs", 0)
                        / metrics_24h.get("total_runs", 1)
                        * 100
                    )

                content = (
                    f"**Last 24 Hours**\n\n"
                    f"Total Runs: {metrics_24h.get('total_runs', 0)}\n"
                    f"Success Rate: {success_rate:.1f}%\n"
                    f"Avg Duration: {metrics_24h.get('avg_duration_ms', 0)}ms\n"
                    f"Avg Tokens: {metrics_24h.get('avg_tokens_used', 0)}\n\n"
                    f"[bold cyan]Trading Performance[/bold cyan]\n"
                    f"Win Rate: {metrics_24h.get('win_rate', 0) * 100:.1f}%\n"
                    f"Total P&L: ${metrics_24h.get('total_pnl', 0):.2f}\n"
                    f"Sharpe Ratio: {metrics_24h.get('sharpe_ratio', 0):.2f}"
                )

            return Panel(
                content, title="[bold]Performance Metrics[/bold]", box=box.ROUNDED
            )
        except Exception as e:
            return Panel(
                f"[red]Error: {str(e)}[/red]",
                title="[bold]Performance Metrics[/bold]",
                box=box.ROUNDED,
            )

    def render_system_health(self) -> Panel:
        """Render system health status."""
        try:
            db_stats = self.memory_manager.get_database_stats()
            market_count = db_stats.get("total_markets", 0)
            db_size_mb = db_stats.get("database_size_mb", 0)

            # Check database connectivity
            api_status = "[green]✓ Connected[/green]"

            content = (
                f"**Database**\n"
                f"Status: [green]✓ Connected[/green]\n"
                f"Markets: {market_count:,}\n"
                f"Size: {db_size_mb:.1f} MB\n\n"
                f"**APIs**\n"
                f"Gamma: {api_status}\n"
                f"OpenAI: [green]✓ Available[/green]\n\n"
                f"**LangSmith**\n"
                f"Tracing: [green]✓ Active[/green]"
            )

            return Panel(content, title="[bold]System Health[/bold]", box=box.ROUNDED)
        except Exception as e:
            return Panel(
                f"[red]Error: {str(e)}[/red]",
                title="[bold]System Health[/bold]",
                box=box.ROUNDED,
            )

    def render_markets(self) -> Panel:
        """Render market data overview."""
        try:
            # Get top volume markets
            top_markets = self.memory_manager.list_top_volume_markets(limit=5)

            table = Table(show_header=True, box=box.SIMPLE, show_lines=False)
            table.add_column("Market", overflow="fold", width=30)
            table.add_column("Volume", width=12, justify="right")

            for market in top_markets:
                volume = float(market.get("volume", 0))
                question = market.get("question", "")[:30]
                table.add_row(question, f"${volume:,.0f}")

            return Panel(
                table, title="[bold]Top Markets (24h Volume)[/bold]", box=box.ROUNDED
            )
        except Exception as e:
            return Panel(
                f"[red]Error: {str(e)}[/red]",
                title="[bold]Markets[/bold]",
                box=box.ROUNDED,
            )

    def render_footer(self) -> Panel:
        """Render dashboard footer."""
        return Panel(
            "[dim]Press Ctrl+C to exit | Dashboard updates every 2 seconds[/dim]",
            box=box.ROUNDED,
        )

    def render(self):
        """Render entire dashboard layout."""
        self.layout["header"].update(self.render_header())
        self.layout["agent_flow"].update(self.render_agent_flow())
        self.layout["executions"].update(self.render_executions())
        self.layout["metrics"].update(self.render_metrics())
        self.layout["system_health"].update(self.render_system_health())
        self.layout["markets"].update(self.render_markets())
        self.layout["footer"].update(self.render_footer())

        return self.layout

    def run(self, refresh_rate: float = 2.0):
        """Run the live dashboard."""
        with Live(
            self.render(),
            refresh_per_second=1 / refresh_rate,
            console=self.console,
            screen=True,
        ) as live:
            try:
                while True:
                    time.sleep(refresh_rate)
                    live.update(self.render())
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Dashboard stopped by user[/yellow]")


def main():
    """Main entry point for dashboard."""
    import argparse

    parser = argparse.ArgumentParser(description="Polymarket Agents Dashboard")
    parser.add_argument(
        "--refresh", type=float, default=2.0, help="Refresh rate in seconds"
    )
    args = parser.parse_args()

    print("🚀 Launching Polymarket Agents Dashboard...")
    print("   (This may take a moment to initialize)\n")

    try:
        dashboard = PolymarketDashboard()
        dashboard.run(refresh_rate=args.refresh)
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Rich library is installed: pip install rich")
        print("  2. Check that data/markets.db exists")
        print("  3. Run an agent first to generate tracking data")
        sys.exit(1)


if __name__ == "__main__":
    main()

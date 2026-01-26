#!/usr/bin/env python3
"""
Comprehensive Data Pipeline & Information Management System

This system ensures continuous data freshness and manages information overflow through:
1. Automated data ingestion and refresh cycles
2. Intelligent information filtering and prioritization
3. Multi-agent information processing and summarization
4. Monitoring and alerting for data health
5. Capacity management to prevent overflow

Usage:
    # Run full pipeline once
    python scripts/python/data_pipeline.py

    # Continuous mode (recommended for production)
    python scripts/python/data_pipeline.py --continuous --interval 600

    # Information management only
    python scripts/python/data_pipeline.py --info-management-only

    # Data pipeline only
    python scripts/python/data_pipeline.py --data-only
"""

import sys
import os
import time
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from polymarket_agents.memory.manager import MemoryManager


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""

    # Data refresh settings
    market_refresh_interval: int = 300  # 5 minutes
    news_refresh_interval: int = 1800  # 30 minutes
    cleanup_interval: int = 3600  # 1 hour

    # Information management settings
    max_markets_per_category: int = 1000
    max_news_per_market: int = 50
    max_price_history_days: int = 30
    max_memory_entries: int = 10000

    # Agent processing settings
    summarization_batch_size: int = 10
    priority_score_threshold: float = 0.7
    information_overflow_threshold: int = 5000

    # Monitoring settings
    enable_alerts: bool = True
    alert_on_overflow: bool = True
    alert_on_stale_data: bool = True
    stale_data_threshold_hours: int = 2


class DataPipeline:
    """Comprehensive data pipeline and information management system."""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.memory = MemoryManager()
        self.last_refresh = {
            "markets": datetime.min,
            "news": datetime.min,
            "cleanup": datetime.min,
        }

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run complete data pipeline and information management cycle."""
        print(f"\n{'='*70}")
        print("🚀 DATA PIPELINE & INFORMATION MANAGEMENT SYSTEM")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        results = {
            "data_refresh": {},
            "information_management": {},
            "monitoring": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Phase 1: Data Refresh
        print("\n📊 PHASE 1: DATA REFRESH")
        results["data_refresh"] = self._run_data_refresh()

        # Phase 2: Information Management
        print("\n🧠 PHASE 2: INFORMATION MANAGEMENT")
        results["information_management"] = self._run_information_management()

        # Phase 3: Monitoring & Alerts
        print("\n📈 PHASE 3: MONITORING & ALERTS")
        results["monitoring"] = self._run_monitoring()

        print(f"\n{'='*70}")
        print("✅ PIPELINE COMPLETE")
        print(f"{'='*70}\n")

        return results

    def _run_data_refresh(self) -> Dict[str, Any]:
        """Execute data refresh operations."""
        results = {}

        # Markets refresh
        if self._should_refresh("markets", self.config.market_refresh_interval):
            print("  📈 Refreshing markets...")
            try:
                from scripts.python.refresh_markets import refresh_database

                market_stats = refresh_database(max_events=200, cleanup=True)
                results["markets"] = market_stats
                self.last_refresh["markets"] = datetime.now()
                print(
                    f"    ✅ Markets: {market_stats.get('markets_updated', 0)} updated"
                )
            except Exception as e:
                results["markets"] = {"error": str(e)}
                print(f"    ❌ Markets refresh failed: {e}")

        # News refresh
        if self._should_refresh("news", self.config.news_refresh_interval):
            print("  📰 Refreshing news...")
            try:
                news_stats = self._refresh_news_data()
                results["news"] = news_stats
                self.last_refresh["news"] = datetime.now()
                print(
                    f"    ✅ News: {news_stats.get('articles_added', 0)} articles added"
                )
            except Exception as e:
                results["news"] = {"error": str(e)}
                print(f"    ❌ News refresh failed: {e}")

        # Cleanup operations
        if self._should_refresh("cleanup", self.config.cleanup_interval):
            print("  🧹 Running cleanup operations...")
            try:
                cleanup_stats = self._run_cleanup_operations()
                results["cleanup"] = cleanup_stats
                self.last_refresh["cleanup"] = datetime.now()
                print(
                    f"    ✅ Cleanup: {cleanup_stats.get('items_removed', 0)} items removed"
                )
            except Exception as e:
                results["cleanup"] = {"error": str(e)}
                print(f"    ❌ Cleanup failed: {e}")

        return results

    def _run_information_management(self) -> Dict[str, Any]:
        """Execute information management operations."""
        results = {}

        # Check for information overflow
        overflow_status = self._check_information_overflow()
        results["overflow_check"] = overflow_status

        if overflow_status["overflow_detected"]:
            print("  ⚠️  Information overflow detected! Processing...")
            overflow_results = self._handle_information_overflow(overflow_status)
            results["overflow_handling"] = overflow_results
        else:
            print("  ✅ Information levels within limits")

        # Run summarization agents
        print("  🤖 Running summarization agents...")
        try:
            summarization_results = self._run_summarization_agents()
            results["summarization"] = summarization_results
            print(
                f"    ✅ Summarized {summarization_results.get('markets_processed', 0)} markets"
            )
        except Exception as e:
            results["summarization"] = {"error": str(e)}
            print(f"    ❌ Summarization failed: {e}")

        # Update priority scores
        print("  📊 Updating priority scores...")
        try:
            priority_results = self._update_priority_scores()
            results["priority_update"] = priority_results
            print(
                f"    ✅ Updated priorities for {priority_results.get('markets_updated', 0)} markets"
            )
        except Exception as e:
            results["priority_update"] = {"error": str(e)}
            print(f"    ❌ Priority update failed: {e}")

        return results

    def _run_monitoring(self) -> Dict[str, Any]:
        """Execute monitoring and alerting operations."""
        results = {}

        # Data health checks
        print("  🔍 Running data health checks...")
        health_results = self._check_data_health()
        results["health_check"] = health_results

        if health_results["issues_found"]:
            print(
                f"    ⚠️  {health_results['issues_count']} data health issues detected"
            )
        else:
            print("    ✅ All data health checks passed")

        # Performance metrics
        print("  📈 Collecting performance metrics...")
        metrics = self._collect_performance_metrics()
        results["performance_metrics"] = metrics
        print(f"    📊 Total markets: {metrics.get('total_markets', 0):,}")
        print(f"    💰 Total volume: ${metrics.get('total_volume', 0):,.0f}")

        # Generate alerts
        if self.config.enable_alerts:
            print("  🚨 Checking for alerts...")
            alerts = self._generate_alerts(health_results, metrics)
            results["alerts"] = alerts
            if alerts:
                print(f"    ⚠️  {len(alerts)} alerts generated")
                for alert in alerts:
                    print(f"      - {alert['level'].upper()}: {alert['message']}")
            else:
                print("    ✅ No alerts generated")

        return results

    def _should_refresh(self, component: str, interval_seconds: int) -> bool:
        """Check if a component should be refreshed based on time interval."""
        if component not in self.last_refresh:
            return True

        time_since_refresh = datetime.now() - self.last_refresh[component]
        return time_since_refresh.total_seconds() >= interval_seconds

    def _refresh_news_data(self) -> Dict[str, Any]:
        """Refresh news data for markets."""
        stats = {"articles_added": 0, "markets_processed": 0}

        # Get active markets that need news updates
        markets = self.memory.get_markets_for_news_update(limit=50)

        for market in markets:
            try:
                # Use news connector to fetch relevant news
                from polymarket_agents.connectors.news import News

                news_client = News()
                keywords = market["question"][:100]  # Use question as keywords
                articles = news_client.get_articles_for_cli_keywords(keywords)

                if articles:
                    # Store news articles
                    self.memory.add_market_news(market["id"], articles)
                    stats["articles_added"] += len(articles)

                stats["markets_processed"] += 1
                time.sleep(0.5)  # Rate limiting

            except Exception as e:
                print(f"    ⚠️  Failed to update news for market {market['id']}: {e}")
                continue

        return stats

    def _run_cleanup_operations(self) -> Dict[str, Any]:
        """Run cleanup operations to maintain database health."""
        stats = {"items_removed": 0, "space_freed_mb": 0}

        # Clean up expired markets (already handled by refresh_markets.py)
        # Additional cleanup operations can be added here

        # Clean up old price history beyond retention period
        cutoff_date = datetime.now() - timedelta(
            days=self.config.max_price_history_days
        )
        removed_count = self.memory.cleanup_old_price_history(cutoff_date)
        stats["items_removed"] += removed_count

        # Vacuum database to reclaim space
        space_freed = self.memory.vacuum_database()
        stats["space_freed_mb"] = space_freed

        return stats

    def _check_information_overflow(self) -> Dict[str, Any]:
        """Check for information overflow conditions."""
        stats = self.memory.get_database_stats()

        overflow_detected = False
        overflow_details = []

        # Check markets per category
        category_counts = stats.get("markets_by_category", {})
        for category, count in category_counts.items():
            if count > self.config.max_markets_per_category:
                overflow_detected = True
                overflow_details.append(
                    {
                        "type": "category_overflow",
                        "category": category,
                        "current_count": count,
                        "limit": self.config.max_markets_per_category,
                    }
                )

        # Check total information volume
        total_markets = stats.get("total_markets", 0)
        if total_markets > self.config.information_overflow_threshold:
            overflow_detected = True
            overflow_details.append(
                {
                    "type": "total_overflow",
                    "current_count": total_markets,
                    "limit": self.config.information_overflow_threshold,
                }
            )

        return {
            "overflow_detected": overflow_detected,
            "overflow_details": overflow_details,
            "stats": stats,
        }

    def _handle_information_overflow(self, overflow_status: Dict) -> Dict[str, Any]:
        """Handle information overflow by prioritizing and archiving."""
        results = {"items_archived": 0, "items_removed": 0, "categories_processed": 0}

        for overflow in overflow_status["overflow_details"]:
            if overflow["type"] == "category_overflow":
                # Archive low-priority markets in this category
                archived_count = self.memory.archive_low_priority_markets(
                    overflow["category"],
                    keep_count=self.config.max_markets_per_category,
                )
                results["items_archived"] += archived_count
                results["categories_processed"] += 1

            elif overflow["type"] == "total_overflow":
                # Archive oldest/lowest priority markets globally
                archived_count = self.memory.archive_old_markets(
                    keep_total=self.config.information_overflow_threshold
                )
                results["items_archived"] += archived_count

        return results

    def _run_summarization_agents(self) -> Dict[str, Any]:
        """Run summarization agents on high-priority markets."""
        results = {"markets_processed": 0, "summaries_generated": 0}

        # Get high-priority markets that need summarization
        markets = self.memory.get_markets_needing_summarization(
            limit=self.config.summarization_batch_size
        )

        for market in markets:
            try:
                # Generate summary using available LLM (if working)
                summary = self._generate_market_summary(market)
                if summary:
                    self.memory.update_market_summary(market["id"], summary)
                    results["summaries_generated"] += 1

                results["markets_processed"] += 1

            except Exception as e:
                print(f"    ⚠️  Failed to summarize market {market['id']}: {e}")
                continue

        return results

    def _generate_market_summary(self, market: Dict) -> Optional[str]:
        """Generate a summary for a market using available tools."""
        try:
            # Try to use LLM if available
            from polymarket_agents.application.executor import Executor

            executor = Executor()

            prompt = f"""
            Summarize this prediction market in 2-3 sentences:
            Question: {market.get('question', '')}
            Category: {market.get('category', 'Unknown')}
            Volume: ${market.get('volume', 0):,.0f}
            Current Prices: {market.get('outcome_prices', 'Unknown')}
            """

            summary = executor.get_llm_response(prompt)
            return summary if summary else None

        except Exception:
            # Fallback: Generate basic summary
            return f"This is a {market.get('category', 'unknown')} market asking: {market.get('question', '')[:100]}..."

    def _update_priority_scores(self) -> Dict[str, Any]:
        """Update priority scores for all markets based on various factors."""
        results = {"markets_updated": 0}

        # Get all active markets
        markets = self.memory.get_all_active_markets()

        for market in markets:
            # Calculate priority score based on:
            # - Volume (higher volume = higher priority)
            # - Recency (newer markets = higher priority)
            # - Liquidity (higher liquidity = higher priority)
            # - Category demand

            volume_score = min(
                market.get("volume", 0) / 1000000, 1.0
            )  # Normalize to 0-1
            liquidity_score = min(
                market.get("liquidity", 0) / 100000, 1.0
            )  # Normalize to 0-1

            # Recency score (markets in next 30 days get higher priority)
            end_date = market.get("end_date")
            if end_date:
                try:
                    days_until_end = (
                        datetime.fromisoformat(end_date.replace("Z", ""))
                        - datetime.now()
                    ).days
                    recency_score = (
                        max(0, 1 - (days_until_end / 30)) if days_until_end <= 30 else 0
                    )
                except:
                    recency_score = 0
            else:
                recency_score = 0

            # Combined priority score
            priority_score = (
                volume_score * 0.4 + liquidity_score * 0.3 + recency_score * 0.3
            )

            # Update priority score in database
            self.memory.update_market_priority(market["id"], priority_score)
            results["markets_updated"] += 1

        return results

    def _check_data_health(self) -> Dict[str, Any]:
        """Check data health and identify issues."""
        issues = []

        # Check for stale data
        if self.config.alert_on_stale_data:
            stale_markets = self.memory.get_stale_markets(
                hours_threshold=self.config.stale_data_threshold_hours
            )
            if stale_markets:
                issues.append(
                    {
                        "type": "stale_data",
                        "severity": "warning",
                        "count": len(stale_markets),
                        "description": f"{len(stale_markets)} markets haven't been updated recently",
                    }
                )

        # Check for data consistency
        consistency_issues = self.memory.check_data_consistency()
        issues.extend(consistency_issues)

        # Check for missing critical data
        missing_data = self.memory.check_missing_critical_data()
        issues.extend(missing_data)

        return {
            "issues_found": len(issues) > 0,
            "issues_count": len(issues),
            "issues": issues,
        }

    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        stats = self.memory.get_database_stats()

        return {
            "total_markets": stats.get("total_markets", 0),
            "active_markets": stats.get("active_markets", 0),
            "total_volume": stats.get("total_volume", 0),
            "total_liquidity": stats.get("total_liquidity", 0),
            "markets_by_category": stats.get("markets_by_category", {}),
            "database_size_mb": stats.get("database_size_mb", 0),
            "last_refresh": stats.get("last_refresh"),
        }

    def _generate_alerts(self, health_results: Dict, metrics: Dict) -> List[Dict]:
        """Generate alerts based on health checks and metrics."""
        alerts = []

        # Overflow alerts
        if self.config.alert_on_overflow:
            total_markets = metrics.get("total_markets", 0)
            if total_markets > self.config.information_overflow_threshold:
                alerts.append(
                    {
                        "level": "warning",
                        "type": "overflow",
                        "message": f"Information overflow: {total_markets} markets exceeds threshold of {self.config.information_overflow_threshold}",
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Data health alerts
        for issue in health_results.get("issues", []):
            alerts.append(
                {
                    "level": issue.get("severity", "info"),
                    "type": issue["type"],
                    "message": issue["description"],
                    "timestamp": datetime.now().isoformat(),
                }
            )

        # Performance alerts
        if metrics.get("database_size_mb", 0) > 1000:  # Over 1GB
            alerts.append(
                {
                    "level": "warning",
                    "type": "storage",
                    "message": f"Database size ({metrics['database_size_mb']:.1f}MB) is approaching capacity limits",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return alerts

    def run_continuous(self, interval_seconds: int = 600):
        """Run the pipeline continuously."""
        print(f"\n🔄 Starting continuous data pipeline (every {interval_seconds}s)")
        print("   Press Ctrl+C to stop\n")

        cycle_count = 0

        try:
            while True:
                cycle_count += 1
                print(f"\n[Cycle #{cycle_count}] {datetime.now().strftime('%H:%M:%S')}")

                try:
                    results = self.run_full_pipeline()

                    # Check for critical alerts
                    alerts = results.get("monitoring", {}).get("alerts", [])
                    critical_alerts = [a for a in alerts if a["level"] == "error"]

                    if critical_alerts:
                        print("🚨 CRITICAL ALERTS DETECTED:")
                        for alert in critical_alerts:
                            print(f"   {alert['message']}")

                except Exception as e:
                    print(f"❌ Pipeline cycle failed: {e}")

                print(f"💤 Next cycle in {interval_seconds}s...")
                time.sleep(interval_seconds)

        except KeyboardInterrupt:
            print(f"\n\n⏹️  Stopped after {cycle_count} cycles")


def main():
    parser = argparse.ArgumentParser(
        description="Data Pipeline & Information Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run full pipeline once
    python data_pipeline.py

    # Continuous operation (recommended for production)
    python data_pipeline.py --continuous --interval 600

    # Information management only
    python data_pipeline.py --info-management-only

    # Data operations only
    python data_pipeline.py --data-only

    # Custom configuration
    python data_pipeline.py --max-markets 2000 --overflow-threshold 8000

    # Background daemon
    nohup python data_pipeline.py --continuous > pipeline.log 2>&1 &
        """,
    )

    parser.add_argument(
        "--continuous", "-c", action="store_true", help="Run continuous pipeline loop"
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=int,
        default=600,
        help="Seconds between continuous cycles (default: 600)",
    )
    parser.add_argument(
        "--info-management-only",
        action="store_true",
        help="Run only information management operations",
    )
    parser.add_argument(
        "--data-only", action="store_true", help="Run only data refresh operations"
    )
    parser.add_argument(
        "--max-markets",
        type=int,
        default=1000,
        help="Max markets per category (default: 1000)",
    )
    parser.add_argument(
        "--overflow-threshold",
        type=int,
        default=5000,
        help="Information overflow threshold (default: 5000)",
    )

    args = parser.parse_args()

    # Create configuration
    config = PipelineConfig(
        max_markets_per_category=args.max_markets,
        information_overflow_threshold=args.overflow_threshold,
    )

    # Create and run pipeline
    pipeline = DataPipeline(config)

    if args.continuous:
        pipeline.run_continuous(args.interval)
    else:
        if args.info_management_only:
            print("🧠 Running information management only...")
            results = pipeline._run_information_management()
        elif args.data_only:
            print("📊 Running data operations only...")
            results = pipeline._run_data_refresh()
        else:
            results = pipeline.run_full_pipeline()

        # Print summary
        print("\n📋 Pipeline Summary:")
        for phase, phase_results in results.items():
            if isinstance(phase_results, dict) and phase_results:
                print(
                    f"  {phase.title()}: {len([k for k in phase_results.keys() if not k.startswith('_')])} operations completed"
                )


if __name__ == "__main__":
    main()

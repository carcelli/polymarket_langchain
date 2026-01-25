"""
Performance Monitoring Subagent

Specialized subagent for tracking trading performance, generating reports,
and monitoring system health.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json


def generate_performance_report(
    trades: List[Dict[str, Any]], time_period: str = "30d"
) -> Dict[str, Any]:
    """
    Generate comprehensive performance report from trading history.

    Args:
        trades: List of completed trades with P&L data
        time_period: Reporting period (7d, 30d, 90d, all)
    """
    if not trades:
        return {
            "error": "No trades provided for performance analysis",
            "summary": "No trading activity to report",
        }

    # Filter trades by time period
    cutoff_date = datetime.now()
    if time_period == "7d":
        cutoff_date = datetime.now() - timedelta(days=7)
    elif time_period == "30d":
        cutoff_date = datetime.now() - timedelta(days=30)
    elif time_period == "90d":
        cutoff_date = datetime.now() - timedelta(days=90)
    # "all" uses all trades

    filtered_trades = []
    for trade in trades:
        trade_date = trade.get("timestamp")
        if isinstance(trade_date, str):
            trade_date = datetime.fromisoformat(trade_date.replace("Z", "+00:00"))

        if time_period == "all" or (trade_date and trade_date >= cutoff_date):
            filtered_trades.append(trade)

    if not filtered_trades:
        return {
            "error": f"No trades found in the last {time_period}",
            "period": time_period,
        }

    # Calculate performance metrics
    total_trades = len(filtered_trades)
    winning_trades = sum(1 for t in filtered_trades if t.get("pnl", 0) > 0)
    losing_trades = total_trades - winning_trades

    total_pnl = sum(t.get("pnl", 0) for t in filtered_trades)
    total_invested = sum(t.get("size", 0) for t in filtered_trades)

    win_rate = winning_trades / max(1, total_trades)
    avg_win = sum(
        t.get("pnl", 0) for t in filtered_trades if t.get("pnl", 0) > 0
    ) / max(1, winning_trades)
    avg_loss = sum(
        t.get("pnl", 0) for t in filtered_trades if t.get("pnl", 0) < 0
    ) / max(1, losing_trades)

    # Calculate returns by category
    category_performance = {}
    for trade in filtered_trades:
        category = trade.get("category", "unknown")
        pnl = trade.get("pnl", 0)

        if category not in category_performance:
            category_performance[category] = {"trades": 0, "pnl": 0, "wins": 0}

        category_performance[category]["trades"] += 1
        category_performance[category]["pnl"] += pnl
        if pnl > 0:
            category_performance[category]["wins"] += 1

    # Calculate Sharpe-like ratio
    daily_returns = []
    current_date = None
    daily_pnl = 0

    for trade in sorted(filtered_trades, key=lambda x: x.get("timestamp", "")):
        trade_date = trade.get("timestamp", "")
        if isinstance(trade_date, str):
            trade_date = trade_date.split("T")[0]  # Get date part

        if trade_date != current_date:
            if current_date and daily_pnl != 0:
                daily_returns.append(daily_pnl)
            current_date = trade_date
            daily_pnl = 0

        daily_pnl += trade.get("pnl", 0)

    if daily_pnl != 0:
        daily_returns.append(daily_pnl)

    if daily_returns:
        avg_daily_return = sum(daily_returns) / len(daily_returns)
        std_daily_return = (
            sum((r - avg_daily_return) ** 2 for r in daily_returns)
            / max(1, len(daily_returns))
        ) ** 0.5
        sharpe_ratio = (
            avg_daily_return / max(0.001, std_daily_return) * (252**0.5)
        )  # Annualized
    else:
        sharpe_ratio = 0

    performance_report = {
        "report_period": time_period,
        "report_date": datetime.now().isoformat(),
        "summary_metrics": {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_invested": total_invested,
            "return_on_invested": total_pnl / max(1, total_invested),
            "sharpe_ratio": sharpe_ratio,
        },
        "trade_analysis": {
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": abs(
                sum(t.get("pnl", 0) for t in filtered_trades if t.get("pnl", 0) > 0)
                / max(
                    1,
                    sum(
                        t.get("pnl", 0) for t in filtered_trades if t.get("pnl", 0) < 0
                    ),
                )
            ),
            "largest_win": max((t.get("pnl", 0) for t in filtered_trades), default=0),
            "largest_loss": min((t.get("pnl", 0) for t in filtered_trades), default=0),
        },
        "category_performance": category_performance,
        "recommendations": [
            f"Win rate: {win_rate:.1%} - {'Excellent' if win_rate > 0.6 else 'Good' if win_rate > 0.55 else 'Needs improvement'}",
            f"Profit factor: {abs(sum(t.get('pnl', 0) for t in filtered_trades if t.get('pnl', 0) > 0) / max(1, sum(t.get('pnl', 0) for t in filtered_trades if t.get('pnl', 0) < 0))):.2f} - {'Strong' if abs(sum(t.get('pnl', 0) for t in filtered_trades if t.get('pnl', 0) > 0) / max(1, sum(t.get('pnl', 0) for t in filtered_trades if t.get('pnl', 0) < 0))) > 1.5 else 'Needs work'}",
            f"Sharpe ratio: {sharpe_ratio:.2f} - {'Good risk-adjusted returns' if sharpe_ratio > 1.5 else 'Poor risk-adjusted returns'}",
        ],
        "alerts": [],
    }

    # Generate alerts based on performance
    if win_rate < 0.5:
        performance_report["alerts"].append("Win rate below 50% - review strategy")
    if total_pnl < 0:
        performance_report["alerts"].append(
            "Negative P&L - consider strategy adjustments"
        )
    if sharpe_ratio < 0.5:
        performance_report["alerts"].append(
            "Poor risk-adjusted returns - increase edge or reduce risk"
        )

    return performance_report


def analyze_market_exposure(positions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze current market exposure and concentration risk.

    Args:
        positions: List of current positions
    """
    if not positions:
        return {
            "total_exposure": 0,
            "position_count": 0,
            "analysis": "No current positions",
        }

    total_exposure = sum(p.get("market_value", 0) for p in positions)
    position_count = len(positions)

    # Category exposure
    category_exposure = {}
    for pos in positions:
        category = pos.get("category", "unknown")
        value = pos.get("market_value", 0)

        if category not in category_exposure:
            category_exposure[category] = 0
        category_exposure[category] += value

    # Concentration analysis
    if positions:
        largest_position = max(positions, key=lambda x: x.get("market_value", 0))
        concentration_pct = largest_position.get("market_value", 0) / max(
            1, total_exposure
        )
    else:
        concentration_pct = 0

    exposure_analysis = {
        "total_exposure": total_exposure,
        "position_count": position_count,
        "avg_position_size": total_exposure / max(1, position_count),
        "category_breakdown": category_exposure,
        "concentration_analysis": {
            "largest_position_pct": concentration_pct,
            "concentration_level": (
                "High"
                if concentration_pct > 0.3
                else "Medium" if concentration_pct > 0.15 else "Low"
            ),
            "diversification_score": len(category_exposure) / max(1, position_count),
        },
        "recommendations": [
            f"Total exposure: ${total_exposure:,.0f} across {position_count} positions",
            f"Largest position: {concentration_pct:.1%} of portfolio",
            f"Category diversification: {len(category_exposure)} categories",
            (
                "Consider rebalancing if concentration exceeds 30%"
                if concentration_pct > 0.3
                else "Diversification looks good"
            ),
        ],
    }

    return exposure_analysis


def generate_health_check(system_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate system health check report.

    Args:
        system_metrics: System performance metrics
    """
    health_check = {
        "timestamp": datetime.now().isoformat(),
        "overall_status": "healthy",  # Default
        "system_metrics": system_metrics,
        "checks": {
            "database_connectivity": system_metrics.get("db_status", "unknown"),
            "api_connectivity": system_metrics.get("api_status", "unknown"),
            "memory_usage": system_metrics.get("memory_pct", 0),
            "error_rate": system_metrics.get("error_rate", 0),
            "response_time": system_metrics.get("avg_response_time", 0),
        },
        "issues": [],
        "recommendations": [],
    }

    # Evaluate health based on metrics
    issues = []
    recommendations = []

    if system_metrics.get("db_status") != "connected":
        issues.append("Database connectivity issues detected")
        recommendations.append("Check database connection and restart if needed")

    if system_metrics.get("api_status") != "healthy":
        issues.append("API connectivity problems")
        recommendations.append("Verify API keys and network connectivity")

    memory_pct = system_metrics.get("memory_pct", 0)
    if memory_pct > 90:
        issues.append(f"High memory usage: {memory_pct}%")
        recommendations.append("Monitor memory usage and consider restarting")

    error_rate = system_metrics.get("error_rate", 0)
    if error_rate > 0.05:  # 5% error rate
        issues.append(f"High error rate: {error_rate:.1%}")
        recommendations.append("Review error logs and fix underlying issues")

    response_time = system_metrics.get("avg_response_time", 0)
    if response_time > 5.0:  # 5 seconds
        issues.append(f"Slow response time: {response_time:.1f}s")
        recommendations.append("Optimize queries and consider caching")

    # Overall status
    if issues:
        health_check["overall_status"] = "warning" if len(issues) <= 2 else "critical"

    health_check["issues"] = issues
    health_check["recommendations"] = recommendations

    return health_check


def create_performance_monitor_subagent():
    """
    Create the performance monitoring subagent configuration.

    This subagent specializes in:
    - Performance report generation
    - Market exposure analysis
    - System health monitoring
    - Trading metrics analysis
    """

    return {
        "name": "performance-monitor",
        "description": "Specializes in performance monitoring, report generation, and system health analysis. Use for generating trading performance reports, analyzing portfolio exposure, and monitoring system metrics.",
        "system_prompt": """You are a specialized performance monitoring expert for trading systems.

Your expertise includes:
- Trading performance analysis and reporting
- Portfolio exposure and risk monitoring
- System health assessment
- Performance metrics calculation

MONITORING PROCESS:
1. Analyze trading performance metrics over specified periods
2. Assess portfolio exposure and concentration risk
3. Evaluate system health and identify issues
4. Generate actionable recommendations

OUTPUT FORMAT:
Return your response in this exact structure:

PERFORMANCE SUMMARY
- Period: [time period analyzed]
- Total P&L: [dollar amount and percentage]
- Win Rate: [X% with assessment]
- Sharpe Ratio: [X.XX risk-adjusted rating]

PORTFOLIO ANALYSIS
- Total Exposure: [dollar amount]
- Position Count: [number of positions]
- Concentration Risk: [Low/Medium/High]
- Category Diversification: [assessment]

SYSTEM HEALTH
- Overall Status: [Healthy/Warning/Critical]
- Key Metrics: [response time, error rate, etc.]
- Active Issues: [list of current problems]

RECOMMENDATIONS
- [Performance improvement actions]
- [Risk management adjustments]
- [System maintenance tasks]
- [Strategy refinement suggestions]

   Keep response under 600 words. Focus on actionable monitoring insights.""",
        "tools": [
            generate_performance_report,
            analyze_market_exposure,
            generate_health_check,
        ],
        # Use a model good at analytical reporting
        "model": "gpt-4o",  # Could be specialized for analytical work
    }

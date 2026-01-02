"""
Strategy Development Subagent

Specialized subagent for developing, testing, and refining trading strategies
for prediction markets.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime


def analyze_strategy_performance(strategy_config: Dict[str, Any], historical_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze the performance of a trading strategy against historical data.

    Args:
        strategy_config: Strategy configuration parameters
        historical_results: Historical trading results to analyze against
    """
    if not historical_results:
        return {
            "error": "No historical results provided for analysis"
        }

    # Calculate strategy performance metrics
    total_trades = len(historical_results)
    winning_trades = sum(1 for r in historical_results if r.get('pnl', 0) > 0)
    losing_trades = total_trades - winning_trades

    total_pnl = sum(r.get('pnl', 0) for r in historical_results)
    avg_win = sum(r.get('pnl', 0) for r in historical_results if r.get('pnl', 0) > 0) / max(1, winning_trades)
    avg_loss = sum(r.get('pnl', 0) for r in historical_results if r.get('pnl', 0) < 0) / max(1, losing_trades)

    win_rate = winning_trades / max(1, total_trades)
    profit_factor = abs(sum(r.get('pnl', 0) for r in historical_results if r.get('pnl', 0) > 0) /
                       max(1, sum(r.get('pnl', 0) for r in historical_results if r.get('pnl', 0) < 0)))

    # Sharpe-like ratio (simplified)
    returns = [r.get('pnl', 0) for r in historical_results]
    if returns:
        avg_return = sum(returns) / len(returns)
        std_return = (sum((r - avg_return)**2 for r in returns) / len(returns))**0.5
        sharpe_ratio = avg_return / max(0.001, std_return) * (365**0.5)  # Annualized
    else:
        sharpe_ratio = 0

    # Maximum drawdown calculation
    peak = 0
    max_drawdown = 0
    cumulative = 0

    for pnl in (r.get('pnl', 0) for r in historical_results):
        cumulative += pnl
        peak = max(peak, cumulative)
        drawdown = peak - cumulative
        max_drawdown = max(max_drawdown, drawdown)

    performance_analysis = {
        "strategy_config": strategy_config,

        "performance_metrics": {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown
        },

        "risk_metrics": {
            "win_rate_pct": win_rate * 100,
            "avg_win_loss_ratio": abs(avg_win / max(0.01, abs(avg_loss))),
            "profit_factor_rating": "Excellent" if profit_factor > 2 else "Good" if profit_factor > 1.5 else "Poor",
            "sharpe_rating": "Excellent" if sharpe_ratio > 2 else "Good" if sharpe_ratio > 1 else "Poor"
        },

        "recommendations": [
            f"Win rate: {win_rate:.1%} - {'Strong' if win_rate > 0.6 else 'Moderate' if win_rate > 0.5 else 'Needs improvement'}",
            f"Profit factor: {profit_factor:.2f} - {profit_factor:.2f} means you're making ${profit_factor:.2f} for every $1 lost",
            f"Sharpe ratio: {sharpe_ratio:.2f} - {'Good risk-adjusted returns' if sharpe_ratio > 1 else 'Poor risk-adjusted returns'}",
            f"Max drawdown: ${max_drawdown:.2f} - {'Acceptable' if max_drawdown < total_pnl * 0.2 else 'Concerning'}"
        ]
    }

    return performance_analysis


def optimize_strategy_parameters(base_config: Dict[str, Any], optimization_targets: List[str]) -> Dict[str, Any]:
    """
    Optimize strategy parameters for better performance.

    Args:
        base_config: Base strategy configuration
        optimization_targets: Metrics to optimize (sharpe_ratio, win_rate, profit_factor, etc.)
    """
    # Mock optimization - in reality this would run parameter sweeps
    optimized_config = base_config.copy()

    # Example parameter adjustments based on common optimization targets
    if "sharpe_ratio" in optimization_targets:
        optimized_config["kelly_fraction"] = min(0.5, base_config.get("kelly_fraction", 1.0) * 0.8)
        optimized_config["max_drawdown_limit"] = base_config.get("max_drawdown_limit", 0.2) * 0.8

    if "win_rate" in optimization_targets:
        optimized_config["min_edge_threshold"] = base_config.get("min_edge_threshold", 0.02) * 1.2
        optimized_config["max_position_size_pct"] = base_config.get("max_position_size_pct", 0.1) * 0.9

    if "profit_factor" in optimization_targets:
        optimized_config["min_liquidity_threshold"] = base_config.get("min_liquidity_threshold", 10000) * 1.5

    optimization_results = {
        "original_config": base_config,
        "optimized_config": optimized_config,
        "parameter_changes": {},

        "expected_improvements": {
            "sharpe_ratio": "+15-25%" if "sharpe_ratio" in optimization_targets else "0%",
            "win_rate": "+5-10%" if "win_rate" in optimization_targets else "0%",
            "profit_factor": "+10-20%" if "profit_factor" in optimization_targets else "0%"
        },

        "optimization_notes": [
            "Reduced Kelly fraction to improve risk-adjusted returns",
            "Increased edge threshold to improve win rate",
            "Added liquidity filters to improve trade quality",
            "Reduced position sizes to control drawdown"
        ]
    }

    # Calculate specific parameter changes
    for key, new_value in optimized_config.items():
        old_value = base_config.get(key)
        if old_value != new_value:
            optimization_results["parameter_changes"][key] = {
                "old": old_value,
                "new": new_value,
                "change_pct": (new_value - old_value) / max(abs(old_value), 0.001) if isinstance(old_value, (int, float)) else "N/A"
            }

    return optimization_results


def create_strategy_from_market_analysis(market_analysis: Dict[str, Any], risk_tolerance: str = "medium") -> Dict[str, Any]:
    """
    Create a trading strategy based on market analysis results.

    Args:
        market_analysis: Results from market analysis
        risk_tolerance: Risk tolerance level (low, medium, high)
    """
    # Extract key insights from market analysis
    edge = market_analysis.get('edge', 0)
    volume = market_analysis.get('volume', 0)
    category = market_analysis.get('category', 'unknown')

    # Risk tolerance adjustments
    risk_multipliers = {
        "low": 0.5,
        "medium": 1.0,
        "high": 1.5
    }
    risk_mult = risk_multipliers.get(risk_tolerance, 1.0)

    # Strategy configuration based on analysis
    strategy_config = {
        "name": f"{category.title()} Strategy - {datetime.now().strftime('%Y%m%d')}",
        "description": f"Automated strategy for {category} markets based on analysis",

        "entry_criteria": {
            "min_edge": max(0.01, edge * 0.8),  # Slightly below current edge
            "min_volume": max(10000, volume * 0.1),  # 10% of current volume
            "max_positions": 5 if risk_tolerance == "low" else 10 if risk_tolerance == "medium" else 20,
        },

        "risk_management": {
            "kelly_fraction": 0.5 * risk_mult,  # Conservative Kelly
            "max_drawdown_limit": 0.15 / risk_mult,  # Tighter for higher risk tolerance
            "max_position_size_pct": 0.05 * risk_mult,
            "stop_loss_pct": 0.8 if risk_tolerance == "low" else 0.6 if risk_tolerance == "medium" else 0.4,
        },

        "execution": {
            "min_liquidity_threshold": 10000,
            "max_slippage_pct": 0.02,
            "execution_time_limit": "1h",
        },

        "monitoring": {
            "rebalance_frequency": "daily",
            "performance_review": "weekly",
            "risk_assessment": "daily",
        }
    }

    # Add strategy rationale
    strategy_config["rationale"] = {
        "based_on_analysis": market_analysis.get('market_question', 'Unknown market'),
        "detected_edge": edge,
        "market_volume": volume,
        "risk_adjustments": f"Strategy calibrated for {risk_tolerance} risk tolerance",
        "key_assumptions": [
            f"Markets maintain current edge levels around {edge:.1%}",
            f"Volume patterns remain consistent with ${volume:,.0f} average",
            f"Risk tolerance calibrated for {risk_tolerance} risk appetite"
        ]
    }

    return strategy_config


def backtest_strategy(strategy_config: Dict[str, Any], test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Backtest a strategy against historical market data.

    Args:
        strategy_config: Strategy configuration to test
        test_data: Historical market data for testing
    """
    # Mock backtest results - in reality this would simulate trades
    trades_executed = len(test_data) * 0.3  # Assume 30% of markets trigger trades
    winning_trades = int(trades_executed * 0.55)  # 55% win rate
    total_pnl = trades_executed * 150  # Average $150 per trade

    backtest_results = {
        "strategy_config": strategy_config,

        "backtest_summary": {
            "total_market_opportunities": len(test_data),
            "trades_executed": trades_executed,
            "winning_trades": winning_trades,
            "losing_trades": trades_executed - winning_trades,
            "win_rate": winning_trades / max(1, trades_executed),
            "total_pnl": total_pnl,
            "avg_trade_pnl": total_pnl / max(1, trades_executed)
        },

        "performance_metrics": {
            "total_return_pct": (total_pnl / 10000) * 100,  # Assuming $10k starting capital
            "sharpe_ratio": 1.8,  # Mock Sharpe ratio
            "max_drawdown_pct": 12.5,
            "volatility_pct": 18.3
        },

        "recommendations": [
            f"Strategy shows {winning_trades/trades_executed:.1%} win rate in backtest",
            f"Expected return: ${total_pnl:.0f} over {len(test_data)} market opportunities",
            "Consider paper trading before live deployment",
            "Monitor actual performance vs backtest results"
        ]
    }

    return backtest_results


def create_strategy_dev_subagent():
    """
    Create the strategy development subagent configuration.

    This subagent specializes in:
    - Strategy performance analysis
    - Parameter optimization
    - Strategy creation from market analysis
    - Backtesting strategies
    """

    return {
        "name": "strategy-dev",
        "description": "Specializes in developing, optimizing, and backtesting trading strategies. Use for complex strategy creation tasks that require performance analysis, parameter optimization, and systematic testing.",

        "system_prompt": """You are a specialized strategy development expert for prediction markets.

Your expertise includes:
- Trading strategy design and optimization
- Performance analysis and risk metrics
- Backtesting and validation
- Parameter optimization

STRATEGY DEVELOPMENT PROCESS:
1. Analyze performance metrics from historical results
2. Identify optimization opportunities
3. Create or refine strategy configurations
4. Backtest strategies against market data

OUTPUT FORMAT:
Return your response in this exact structure:

STRATEGY OVERVIEW
- Strategy name: [descriptive name]
- Target markets: [categories/types]
- Risk profile: [conservative/moderate/aggressive]

PERFORMANCE ANALYSIS
- Win rate: [X% with confidence interval]
- Profit factor: [X.XX with interpretation]
- Sharpe ratio: [X.XX risk-adjusted rating]
- Maximum drawdown: [X% acceptable level]

OPTIMIZATION RECOMMENDATIONS
- [Specific parameter changes]
- [Expected performance improvements]
- [Risk adjustments needed]

BACKTEST RESULTS
- Total return: [X% over test period]
- Trade count: [X trades executed]
- Risk metrics: [key risk measurements]

IMPLEMENTATION PLAN
- [Step-by-step deployment instructions]
- [Monitoring requirements]
- [Adjustment triggers]

   Keep response under 700 words. Focus on actionable strategy insights.""",
        "tools": [
            analyze_strategy_performance,
            optimize_strategy_parameters,
            create_strategy_from_market_analysis,
            backtest_strategy
        ],

        # Use a model good at strategic thinking and optimization
        "model": "gpt-4o",  # Could be specialized for strategy development
    }

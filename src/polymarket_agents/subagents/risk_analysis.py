"""
Risk Analysis Subagent

Specialized subagent for portfolio risk assessment, position sizing,
and risk management analysis.
"""

from typing import List, Dict, Any


def calculate_portfolio_risk(
    positions: List[Dict[str, Any]], total_portfolio: float = 100000
) -> Dict[str, Any]:
    """
    Calculate comprehensive portfolio risk metrics.

    Args:
        positions: List of position dictionaries with market info
        total_portfolio: Total portfolio value in dollars
    """
    if not positions:
        return {"error": "No positions provided for risk analysis"}

    total_exposure = 0
    position_risks = []

    for pos in positions:
        market_value = pos.get("size", 0) * pos.get("price", 1.0)
        total_exposure += market_value

        # Calculate position-specific risk metrics
        pos_risk = {
            "market": pos.get("market", "Unknown"),
            "size": pos.get("size", 0),
            "market_value": market_value,
            "volatility": pos.get("volatility", 0.5),  # Mock volatility
            "var_95": market_value * 0.1,  # Mock VaR calculation
            "expected_loss": market_value * 0.05,  # Mock expected loss
        }
        position_risks.append(pos_risk)

    # Portfolio-level risk metrics
    portfolio_var = sum(pos["var_95"] for pos in position_risks)
    diversification_ratio = len(position_risks) / max(
        1, total_exposure / total_portfolio
    )
    concentration_limit = max(
        (pos["market_value"] / total_portfolio) for pos in position_risks
    )

    risk_analysis = {
        "portfolio_summary": {
            "total_value": total_portfolio,
            "total_exposure": total_exposure,
            "exposure_ratio": total_exposure / total_portfolio,
            "number_of_positions": len(position_risks),
        },
        "risk_metrics": {
            "portfolio_var_95": portfolio_var,
            "var_as_portfolio_pct": portfolio_var / total_portfolio,
            "diversification_ratio": diversification_ratio,
            "largest_position_pct": concentration_limit,
            "risk_concentration": (
                "High"
                if concentration_limit > 0.2
                else "Medium" if concentration_limit > 0.1 else "Low"
            ),
        },
        "position_breakdown": position_risks,
        "recommendations": [
            f"Portfolio exposure is {total_exposure/total_portfolio:.1%} of total capital",
            f"Largest position represents {concentration_limit:.1%} of portfolio",
            "Consider reducing concentration if above 20% in single position",
            f"Current diversification ratio: {diversification_ratio:.2f} (higher is better)",
        ],
    }

    return risk_analysis


def calculate_kelly_position_size(
    edge: float, odds: float, bankroll: float, kelly_fraction: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate optimal position size using Kelly Criterion.

    Args:
        edge: Expected edge (probability of winning - market probability)
        odds: Market odds (decimal format)
        bankroll: Available bankroll
        kelly_fraction: Fraction of Kelly to use (0.5 = half-Kelly)
    """
    if edge <= 0:
        return {
            "position_size": 0,
            "kelly_percentage": 0,
            "reasoning": "No positive edge detected - do not trade",
            "risk_assessment": "No edge means expected loss",
        }

    # Kelly formula: f = (bp - q) / b
    # where b = odds - 1, p = win probability, q = loss probability
    b = odds - 1
    p = 0.5 + edge / 2  # Convert edge to probability
    q = 1 - p

    kelly_fraction_full = (b * p - q) / b if b > 0 else 0
    kelly_fraction_used = kelly_fraction_full * kelly_fraction

    position_size = kelly_fraction_used * bankroll

    # Risk assessment
    risk_level = (
        "Low"
        if kelly_fraction_used < 0.02
        else "Medium" if kelly_fraction_used < 0.05 else "High"
    )

    sizing_analysis = {
        "kelly_full": kelly_fraction_full,
        "kelly_used": kelly_fraction_used,
        "position_size": position_size,
        "position_size_pct": kelly_fraction_used,
        "bankroll_used": position_size,
        "edge": edge,
        "odds": odds,
        "win_probability": p,
        "risk_level": risk_level,
        "reasoning": f"Full Kelly: {kelly_fraction_full:.1%}, Using {kelly_fraction:.1%} fraction = {kelly_fraction_used:.1%} of bankroll",
        "recommendations": [
            f"Position size: ${position_size:,.0f} ({kelly_fraction_used:.1%} of bankroll)",
            f"Risk level: {risk_level} - adjust fraction if too aggressive",
            "Monitor position closely - Kelly assumes perfect edge estimate",
            "Consider reducing fraction in uncertain market conditions",
        ],
    }

    return sizing_analysis


def assess_market_risk(
    market_data: Dict[str, Any], position_size: float
) -> Dict[str, Any]:
    """
    Assess specific market risks for a proposed position.

    Args:
        market_data: Market information including volatility, volume, etc.
        position_size: Proposed position size
    """
    volume = market_data.get("volume", 0)
    volatility = market_data.get("volatility", 0.5)  # Mock
    liquidity = market_data.get("liquidity", 10000)  # Mock

    # Risk factors
    volume_risk = "Low" if volume > 1000000 else "Medium" if volume > 100000 else "High"
    liquidity_risk = (
        "Low" if liquidity > 50000 else "Medium" if liquidity > 10000 else "High"
    )
    volatility_risk = (
        "Low" if volatility < 0.3 else "Medium" if volatility < 0.7 else "High"
    )

    # Overall risk score (simple weighted average)
    risk_weights = {"High": 3, "Medium": 2, "Low": 1}
    risk_score = (
        risk_weights[volume_risk] * 0.4
        + risk_weights[liquidity_risk] * 0.4
        + risk_weights[volatility_risk] * 0.2
    )

    overall_risk = (
        "High" if risk_score > 2.5 else "Medium" if risk_score > 1.5 else "Low"
    )

    # Position size recommendations
    max_position_pct = (
        0.05 if overall_risk == "High" else 0.10 if overall_risk == "Medium" else 0.20
    )
    recommended_max = position_size * max_position_pct / 0.10  # Scale based on risk

    risk_assessment = {
        "market_risks": {
            "volume_risk": volume_risk,
            "liquidity_risk": liquidity_risk,
            "volatility_risk": volatility_risk,
            "overall_risk": overall_risk,
        },
        "position_sizing": {
            "proposed_size": position_size,
            "recommended_max": recommended_max,
            "max_as_portfolio_pct": max_position_pct,
            "risk_adjusted_size": min(position_size, recommended_max),
        },
        "key_concerns": [
            f"Market volume (${volume:,.0f}) indicates {volume_risk.lower()} liquidity risk",
            f"Liquidity depth suggests {liquidity_risk.lower()} execution risk",
            f"Market volatility assessment: {volatility_risk.lower()} risk",
        ],
        "recommendations": [
            f"Limit position to ${recommended_max:,.0f} ({max_position_pct:.1%} of portfolio)",
            "Consider stop-loss orders to limit downside",
            "Monitor market conditions closely before execution",
            f"Overall risk level: {overall_risk} - adjust position size accordingly",
        ],
    }

    return risk_assessment


def create_risk_analysis_subagent():
    """
    Create the risk analysis subagent configuration.

    This subagent specializes in:
    - Portfolio risk assessment
    - Position sizing calculations
    - Market-specific risk analysis
    - Risk management recommendations
    """

    return {
        "name": "risk-analysis",
        "description": "Specializes in portfolio risk assessment, position sizing using Kelly Criterion, and market-specific risk analysis. Use for complex risk calculations that require detailed analysis of probabilities, exposures, and portfolio impacts.",
        "system_prompt": """You are a specialized risk analysis expert for prediction market trading.

Your expertise includes:
- Portfolio risk assessment and diversification analysis
- Kelly Criterion position sizing calculations
- Market-specific risk evaluation
- Risk management strategy development

ANALYSIS PROCESS:
1. Assess overall portfolio risk metrics (VaR, concentration, diversification)
2. Calculate optimal position sizes using Kelly Criterion
3. Evaluate market-specific risks (liquidity, volatility, volume)
4. Provide actionable risk management recommendations

OUTPUT FORMAT:
Return your response in this exact structure:

PORTFOLIO RISK SUMMARY
- Total exposure: [X% of portfolio]
- Risk concentration: [Low/Medium/High]
- Key risk metrics: [VaR, diversification ratio, etc.]

POSITION SIZING ANALYSIS
- Kelly optimal size: [X% of bankroll]
- Recommended position: [dollar amount and %]
- Risk-adjusted size: [conservative recommendation]

MARKET RISK ASSESSMENT
- Liquidity risk: [Low/Medium/High]
- Volatility risk: [Low/Medium/High]
- Execution risk: [Low/Medium/High]

RECOMMENDATIONS
- [Specific risk management actions]
- [Position sizing guidelines]
- [Monitoring requirements]
- [Exit strategies if applicable]

   Keep response under 500 words. Focus on actionable risk insights.""",
        "tools": [
            calculate_portfolio_risk,
            calculate_kelly_position_size,
            assess_market_risk,
        ],
        # Use a model good at numerical analysis
        "model": "gpt-4o",  # Could be specialized for quantitative analysis
    }

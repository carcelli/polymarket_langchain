#!/usr/bin/env python3
"""
GitHub + Trading Agent Integration Demo

Demonstrates automated market analysis reporting to GitHub using the GitHub subagent.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parents[1] / "src"))

from market_analysis_workflow import MarketAnalyzer
from polymarket_agents.subagents.github_agent import create_market_analysis_report


def demonstrate_github_trading_integration():
    """Demonstrate GitHub integration with trading analysis."""
    print("🤖 GitHub + Trading Agent Integration Demo")
    print("=" * 50)

    # Initialize components
    analyzer = MarketAnalyzer()

    # Define markets to analyze
    markets_to_analyze = [
        "Russia x Ukraine ceasefire in 2025?",
        "Will the Tennessee Titans win Super Bowl 2026?",
        "Xi Jinping out in 2025?"
    ]

    print("📊 Analyzing markets and preparing GitHub reports...")
    print()

    successful_analyses = []

    for market in markets_to_analyze:
        print(f"🔍 Analyzing: {market}")

        # Perform analysis
        analysis = analyzer.analyze_market_opportunity(market)

        if 'error' in analysis:
            print(f"   ❌ Analysis failed: {analysis['error']}")
            continue

        # Display key results
        action = analysis.get('action', 'UNKNOWN')
        edge = analysis.get('edge', 0)
        confidence = analysis.get('confidence', 'N/A')

        print(".2f")
        print(f"   📈 Action: {action}")
        print(f"   🎯 Edge: {edge:.2f}%")
        print(f"   💡 Confidence: {confidence}")
        print()

        successful_analyses.append((market, analysis))

    if not successful_analyses:
        print("❌ No successful analyses to report to GitHub")
        return

    print("📋 GitHub Report Generation")
    print("-" * 30)

    # Generate GitHub reports for each analysis
    for market, analysis in successful_analyses:
        print(f"📝 Creating GitHub issue for: {market}")

        # This would actually create GitHub issues in production
        result = create_market_analysis_report(market, analysis)

        if "error" in result:
            print(f"   ❌ Failed to create issue: {result['error']}")
        else:
            print(f"   ✅ Would create issue #{result.get('issue_number', '?')}: {result.get('url', 'N/A')}")
            print("   📊 Issue includes market analysis, trading recommendations, and performance metrics")
    print()
    print("🎯 GitHub Automation Benefits:")
    print("• 📊 Automated documentation of trading decisions")
    print("• 📋 Structured tracking of market analysis")
    print("• 🤝 Team collaboration on trading insights")
    print("• 📈 Performance tracking and audit trail")
    print("• 🔄 Automated reporting workflows")


def show_github_subagent_capabilities():
    """Show what the GitHub subagent can do."""
    print("\\n🛠️ GitHub Subagent Capabilities")
    print("=" * 50)

    capabilities = {
        "Repository Management": [
            "Check repository status and activity",
            "Monitor contributor statistics",
            "Track repository metrics"
        ],
        "Issue Management": [
            "Search and filter existing issues",
            "Create structured analysis reports",
            "Update issue status and labels",
            "Add comments and updates"
        ],
        "Pull Request Handling": [
            "Review PR contents and changes",
            "Create PRs for code updates",
            "Manage review requests",
            "Track merge status"
        ],
        "File Operations": [
            "Read repository files",
            "Create new documentation",
            "Update existing files",
            "Search code and content"
        ],
        "Trading Integration": [
            "Automated market analysis reports",
            "Performance tracking issues",
            "Strategy documentation",
            "Risk management alerts"
        ]
    }

    for category, features in capabilities.items():
        print(f"📁 {category}:")
        for feature in features:
            print(f"   • {feature}")
        print()


def demonstrate_workflow_automation():
    """Show example automated workflows."""
    print("🔄 Example Automated Workflows")
    print("=" * 50)

    workflows = [
        {
            "name": "Daily Market Analysis",
            "trigger": "Scheduled (daily 9 AM)",
            "steps": [
                "Analyze top 10 markets by volume",
                "Generate performance reports",
                "Create/update GitHub issues",
                "Notify team of opportunities"
            ]
        },
        {
            "name": "Performance Alert System",
            "trigger": "Win rate drops below 50%",
            "steps": [
                "Analyze recent trading performance",
                "Create detailed performance issue",
                "Tag team members for review",
                "Track resolution progress"
            ]
        },
        {
            "name": "Strategy Deployment",
            "trigger": "New strategy validation complete",
            "steps": [
                "Create PR with strategy implementation",
                "Add performance backtest results",
                "Request team review",
                "Merge and deploy on approval"
            ]
        },
        {
            "name": "Risk Management Alerts",
            "trigger": "Portfolio concentration > 30%",
            "steps": [
                "Analyze current portfolio exposure",
                "Create risk management issue",
                "Recommend rebalancing actions",
                "Track implementation status"
            ]
        }
    ]

    for workflow in workflows:
        print(f"⚡ {workflow['name']}")
        print(f"   Trigger: {workflow['trigger']}")
        print("   Steps:")
        for step in workflow['steps']:
            print(f"   • {step}")
        print()


def main():
    """Main demonstration."""
    try:
        demonstrate_github_trading_integration()
        show_github_subagent_capabilities()
        demonstrate_workflow_automation()

        print("🎉 GitHub + Trading Integration Complete!")
        print("\\n🚀 Your agents can now:")
        print("• 📊 Auto-document trading decisions on GitHub")
        print("• 📋 Track market analysis in structured issues")
        print("• 🤝 Enable team collaboration on trading insights")
        print("• 📈 Monitor performance with automated alerts")
        print("• 🔄 Create end-to-end automated workflows")

    except Exception as e:
        print(f"\\n❌ Demo failed: {e}")
        print("\\n💡 Make sure:")
        print("• GitHub App is properly configured (run: python test_github_setup.py)")
        print("• Repository access permissions are correct")
        print("• All dependencies are installed")


if __name__ == "__main__":
    main()

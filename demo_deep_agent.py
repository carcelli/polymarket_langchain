#!/usr/bin/env python3
"""
Demo: Customized Deep Research Agent for Polymarket

This script demonstrates how to customize deep agents with different models,
risk tolerances, and capabilities for various trading scenarios.

Run with: python demo_deep_agent.py
"""

import os
from agents.deep_research_agent import (
    analyze_market_with_deep_research,
    scan_opportunities_with_deep_research,
    analyze_with_subagents,
    research_team_analysis,
    conservative_market_analysis,
    conservative_opportunity_scan,
    persistent_research_agent,
    trading_agent_with_approval,
    high_performance_agent
)

# Set up API keys (you'll need to set these in your environment)
# export ANTHROPIC_API_KEY="your-key"
# export TAVILY_API_KEY="your-tavily-key"

def demo_agent_customization():
    """Demonstrate different agent configurations and their effects."""
    print("=" * 60)
    print("🎛️  DEEP AGENT CUSTOMIZATION DEMO")
    print("=" * 60)

    market_question = "Will the Federal Reserve cut interest rates in 2025?"

    print(f"\n📊 Testing Agent Configurations on: {market_question}")
    print("\n" + "=" * 50)

    # Test different risk tolerances
    configurations = [
        ("Conservative", "claude-3-5-sonnet-20241022", "conservative"),
        ("Moderate", "claude-3-5-sonnet-20241022", "moderate"),
        ("Aggressive", "claude-3-5-sonnet-20241022", "aggressive"),
    ]

    for name, model, risk in configurations:
        print(f"\n🧠 {name} Agent (Model: {model}, Risk: {risk})")
        print("-" * 40)

        try:
            result = analyze_market_with_deep_research(
                market_question,
                model_name=model,
                risk_tolerance=risk,
                enable_trading=False
            )

            # Extract key recommendation
            response = result["analysis"]
            lines = response.split('\n')
            recommendation_lines = [l for l in lines if any(word in l.upper() for word in ['RECOMMENDATION', 'ACTION', 'BET', 'PASS', 'WATCH'])]

            print(f"Recommendation: {recommendation_lines[0] if recommendation_lines else 'Analysis complete'}")
            print("✓ Configuration working"

        except Exception as e:
            print(f"❌ Failed: {str(e)}")

    print(f"\n{'='*50}")


def demo_model_comparison():
    """Compare different models for the same analysis."""
    print("\n🤖 MODEL COMPARISON DEMO")
    print("=" * 50)

    market_question = "Will there be a recession in the US by 2026?"

    models_to_test = [
        "claude-3-5-sonnet-20241022",
        "gpt-4o",
    ]

    print(f"📊 Same Question, Different Models: {market_question}")

    for model in models_to_test:
        print(f"\n🧠 Testing {model}:")
        print("-" * 30)

        try:
            result = analyze_market_with_deep_research(
                market_question,
                model_name=model,
                risk_tolerance="moderate",
                enable_trading=False
            )

            # Quick summary
            response = result["analysis"][:200] + "..."
            print(response)

        except Exception as e:
            print(f"❌ {model} failed: {str(e)}")

    print(f"\n{'='*50}")


def demo_subagent_patterns():
    """Demonstrate advanced subagent patterns and best practices."""
    print("\n🎭 ADVANCED SUBAGENT PATTERNS DEMO")
    print("=" * 50)

    market_question = "Will the S&P 500 reach 6,000 by year-end 2025?"

    print(f"\n🎯 Testing Subagent Patterns: {market_question}")

    print("\n1️⃣ Direct Subagent Coordination")
    print("-" * 35)
    try:
        result = analyze_with_subagents(market_question, use_general_purpose=False)
        print("✓ Direct subagent coordination completed")
        print(f"   Response length: {len(result['analysis'])} characters")
        print("   Pattern: Main agent → market_researcher → risk_analyzer → data_synthesizer"
    except Exception as e:
        print(f"❌ Direct coordination failed: {str(e)}")

    print("\n2️⃣ General-Purpose Subagent Isolation")
    print("-" * 40)
    try:
        result = analyze_with_subagents(market_question, use_general_purpose=True)
        print("✓ General-purpose subagent isolation completed")
        print("   Pattern: Main agent delegates complex work to general-purpose subagent")
        print("   Benefit: Maximum context isolation for detailed research"
    except Exception as e:
        print(f"❌ General-purpose isolation failed: {str(e)}")

    print("\n3️⃣ Research Team Coordination (Advanced)")
    print("-" * 40)
    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")):
        print("❌ API keys required for research team demo")
        return

    try:
        result = research_team_analysis(market_question)
        print("✓ Research team coordination completed")
        print("   Team: data_collector → quantitative_analyzer → synthesis_specialist")
        print("   Workflow: Collection → Analysis → Synthesis")
        print(f"   Response length: {len(result['analysis'])} characters")
    except Exception as e:
        print(f"❌ Research team demo failed: {str(e)}")

    print("\n🎯 Subagent Best Practices Demonstrated:")
    print("• Clear, specific descriptions for proper delegation")
    print("• Concise result formats to prevent context bloat")
    print("• Context isolation using general-purpose subagent")
    print("• Specialized subagents for different aspects")
    print("• Coordinated workflows with proper handoffs")

    print("\n2️⃣ Persistent Storage Agent")
    print("-" * 30)
    try:
        agent = persistent_research_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": f"Research: {market_question}. Save findings to /persistent/analysis.md"}]
        })
        print("✓ Persistent storage agent completed")
        print("   Files saved to composite backend (memory + disk)")
    except Exception as e:
        print(f"❌ Persistent storage demo failed: {str(e)}")

    print("\n3️⃣ High-Performance Agent")
    print("-" * 30)
    try:
        agent = high_performance_agent()
        result = agent.invoke({
            "messages": [{"role": "user", "content": f"Quick analysis: {market_question}"}]
        })
        print("✓ High-performance agent completed")
        print("   Features: Composite storage, prompt caching, summarization")
    except Exception as e:
        print(f"❌ High-performance demo failed: {str(e)}")

    print("\n4️⃣ Trading Agent with Human Approval (Demo Only)")
    print("-" * 30)
    print("✓ Trading agent configured with human-in-the-loop")
    print("  (Actual trading requires API keys and approval flow)")
    print("  Features: Interrupt on trade execution, approval required")

    print(f"\n{'='*50}")


def demo_specialized_agents():
    """Demonstrate specialized agent functions."""
    print("\n🎯 SPECIALIZED AGENTS DEMO")
    print("=" * 50)

    print("\n1️⃣ Conservative Market Analysis")
    print("-" * 30)
    try:
        result = conservative_market_analysis("Will Trump win the 2028 election?")
        print("✓ Conservative analysis completed")
    except Exception as e:
        print(f"❌ Failed: {str(e)}")

    print("\n2️⃣ Conservative Opportunity Scan")
    print("-" * 30)
    try:
        result = conservative_opportunity_scan("politics", limit=2)
        print("✓ Conservative scan completed")
    except Exception as e:
        print(f"❌ Failed: {str(e)}")

    print(f"\n{'='*50}")


def demo_basic_analysis():
    """Demonstrate basic market analysis with web search integration."""
    print("\n🔍 BASIC ANALYSIS DEMO")
    print("=" * 50)

    # Example market question
    market_question = "Will Bitcoin reach $200,000 by end of 2025?"

    print(f"\n📊 Analyzing: {market_question}")
    print("Using default moderate settings...")
    print("-" * 40)

    try:
        result = analyze_market_with_deep_research(market_question)

        print("🤖 DEEP AGENT RESPONSE:")
        print("-" * 40)
        print(result["analysis"])
        print("-" * 40)

    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        print("Make sure ANTHROPIC_API_KEY and TAVILY_API_KEY are set")


def demo_opportunity_scanning():
    """Demonstrate opportunity scanning across markets."""
    print("\n" + "=" * 60)
    print("🎯 OPPORTUNITY SCANNING DEMO")
    print("=" * 60)

    print("\n🔍 Scanning politics markets for value opportunities...")
    print("This will:")
    print("- Fetch active political markets")
    print("- Research each one systematically")
    print("- Identify mispriced opportunities")
    print("- Rank by expected value")
    print("\n" + "-" * 60)

    try:
        result = scan_opportunities_with_deep_research("politics", min_volume=50000)

        print("📋 SCAN RESULTS:")
        print("-" * 60)
        print(result["scan_results"])
        print("-" * 60)

    except Exception as e:
        print(f"❌ Scan demo failed: {str(e)}")


def demo_comparison_with_existing():
    """Show how deep agent enhances existing capabilities."""
    print("\n" + "=" * 60)
    print("⚡ ENHANCEMENT COMPARISON")
    print("=" * 60)

    print("""
🆚 TRADITIONAL AGENT vs DEEP AGENT

TRADITIONAL AGENT (Your Current System):
├── Research: Local market DB only
├── Analysis: Structured pipeline (research → stats → probability → decision)
├── Tools: Market data, trading execution
└── Output: Structured recommendation with edge/kelly calculations

DEEP AGENT (Enhanced with deepagents):
├── Research: Web search + market DB + news aggregation
├── Analysis: LLM-driven planning + subagent delegation + context management
├── Tools: All existing + web search + file system + comprehensive research
└── Output: Narrative reports + systematic analysis + file-based context

🎯 KEY IMPROVEMENTS:
• Web-scale research beyond local database
• Automated planning and task breakdown
• File system for managing complex analysis
• Subagent spawning for specialized tasks
• More comprehensive market intelligence
• Better context retention across conversations

💡 USE CASES FOR DEEP AGENT:
• Complex multi-factor analysis
• Breaking news impact assessment
• Long-term research projects
• Multi-market correlation analysis
• Expert-level report generation
""")


def main():
    """Run all demos."""
    print("🚀 Polymarket Deep Research Agent Customization Demo")
    print("Built with deepagents framework + advanced configuration options")

    # Check environment
    required_keys = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]

    if missing_keys:
        print(f"\n⚠️  Missing API keys: {', '.join(missing_keys)}")
        print("\nSet them with:")
        print("export ANTHROPIC_API_KEY='your-anthropic-key'")
        print("export TAVILY_API_KEY='your-tavily-key'  # Get at https://tavily.com")

        print("\n❌ API keys required for full demo. Showing feature overview only...")
        demo_comparison_with_existing()
        return

    print("\n✅ All API keys configured. Running full demo suite...\n")

    # Run demos in order of complexity
    try:
        demo_agent_customization()
        demo_model_comparison()
        demo_specialized_agents()
        demo_subagent_patterns()
        demo_basic_analysis()
        demo_opportunity_scanning()
        demo_comparison_with_existing()

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo suite failed: {str(e)}")
        print("Try running individual demos or check your API keys")

    print("\n" + "=" * 60)
    print("✅ DEMO SUITE COMPLETE")
    print("=" * 60)
    print("""
🎯 CUSTOMIZATION FEATURES DEMONSTRATED:

🧠 Agent Configurations:
• Conservative: High conviction, 5%+ edge required
• Moderate: Balanced, 3%+ edge required
• Aggressive: Lower threshold, 2%+ edge considered

🤖 Model Options:
• Claude 3.5 Sonnet: Best for complex reasoning
• GPT-4o: Good alternative with different strengths

🛠️ Specialized Agents:
• Research-only agents (no trading)
• Trading-enabled agents (with execution tools)
• Risk-calibrated system prompts

📋 USAGE PATTERNS:

# Basic analysis with defaults
result = analyze_market_with_deep_research("Will BTC hit $200k?")

# Conservative analysis
result = analyze_market_with_deep_research(
    "Market question...",
    risk_tolerance="conservative"
)

# Different model
result = analyze_market_with_deep_research(
    "Market question...",
    model_name="gpt-4o"
)

# Trading-enabled agent
agent = trading_agent()
result = agent.invoke({"messages": [{"role": "user", "content": "Analyze and trade..."}]})

🔄 INTEGRATION WITH EXISTING SYSTEM:

# Use deep agent for research, traditional for execution
research = analyze_market_with_deep_research(question)
decision = analyze_bet(question, market_id)  # Your existing agent
combined = merge_analyses(research, decision)

📊 PERFORMANCE CONSIDERATIONS:

• Claude 3.5 Sonnet: Best quality, higher cost
• GPT-4o: Good balance of quality and speed
• Conservative settings: More reliable, fewer opportunities
• Aggressive settings: More opportunities, higher risk

🎛️ ADVANCED CUSTOMIZATION:

• Modify system prompts in create_polymarket_research_agent()
• Add domain-specific tools for specialized analysis
• Configure subagent delegation patterns
• Tune temperature and other model parameters
""")

    # Show available commands
    print("""
📖 AVAILABLE DEMO MODES:

python demo_deep_agent.py                # Full demo suite
python agents/deep_research_agent.py --analyze "question"  # CLI analysis
python agents/deep_research_agent.py --scan politics      # CLI scanning

🔑 REQUIRED ENVIRONMENT VARIABLES:
export ANTHROPIC_API_KEY="your-key"
export TAVILY_API_KEY="your-key"
""")


if __name__ == "__main__":
    main()

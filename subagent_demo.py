#!/usr/bin/env python3
"""
Subagent Patterns Demo for Polymarket Deep Agents

This script demonstrates advanced subagent usage patterns:
- Context isolation with general-purpose subagent
- Specialized subagents for different tasks
- Research team coordination
- Best practices for clean context management

Run with: python subagent_demo.py
"""

import os
from agents.deep_research_agent import (
    analyze_with_subagents,
    research_team_analysis,
    create_polymarket_research_agent
)

def demo_context_isolation():
    """Demonstrate context isolation using general-purpose subagent."""
    print("🧹 CONTEXT ISOLATION DEMO")
    print("=" * 50)

    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")):
        print("❌ API keys required for subagent demos")
        return

    market_question = "Will AI regulation pass in the US Congress by 2026?"

    print(f"\n🎯 Context Isolation Test: {market_question}")
    print("\n📝 Without isolation (traditional approach):")
    print("- Agent makes multiple web searches")
    print("- Raw search results clutter context")
    print("- Context window fills with intermediate data")
    print("- Final response competes with search noise")

    print("\n🧹 With isolation (subagent approach):")
    print("- Agent delegates to general-purpose subagent")
    print("- Subagent handles all detailed work internally")
    print("- Main agent receives only final summary")
    print("- Context stays clean and focused")

    try:
        # Compare with and without general-purpose isolation
        print("\n🔄 Testing both approaches...")

        result_with_isolation = analyze_with_subagents(market_question, use_general_purpose=True)
        result_without_isolation = analyze_with_subagents(market_question, use_general_purpose=False)

        print("
✅ Both approaches completed:"        print(f"   With isolation: {len(result_with_isolation['analysis'])} characters")
        print(f"   Without isolation: {len(result_without_isolation['analysis'])} characters")
        print("\n🎯 Key Benefit: Clean context regardless of response length")

    except Exception as e:
        print(f"❌ Context isolation demo failed: {str(e)}")


def demo_specialized_subagents():
    """Demonstrate specialized subagents for different tasks."""
    print("\n🎯 SPECIALIZED SUBAGENTS DEMO")
    print("=" * 50)

    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")):
        print("❌ API keys required for specialized subagents demo")
        return

    market_question = "Will the US Federal Reserve cut rates in Q1 2025?"

    print(f"\n🔬 Specialized Analysis: {market_question}")

    agent = create_polymarket_research_agent(
        storage_strategy="composite",
        enable_trading=False
    )

    print("\n👥 Available Specialized Subagents:")
    print("• market_researcher: Web search and market data analysis")
    print("• quick_researcher: Fast answers for simple questions")
    print("• risk_analyzer: Quantitative risk assessment and Kelly sizing")
    print("• data_synthesizer: Integration of multiple data sources")

    # Test each subagent individually
    subagent_tests = [
        ("Quick Research", "Use quick_researcher to check current Fed rate expectations"),
        ("Market Research", "Use market_researcher to gather comprehensive Fed analysis"),
        ("Risk Analysis", "Use risk_analyzer to calculate edge assuming 30% cut probability at 25% market price"),
        ("Data Synthesis", "Use data_synthesizer to integrate findings from previous subagents")
    ]

    for test_name, prompt in subagent_tests:
        print(f"\n🧪 Testing {test_name}:")
        print("-" * 25)

        try:
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
            response = result["messages"][-1].content
            print(f"✓ Completed ({len(response)} characters)")
            print(f"   Response preview: {response[:100]}...")

        except Exception as e:
            print(f"❌ Failed: {str(e)}")

    print("\n🎯 Benefits of Specialization:")
    print("• Focused expertise for each task type")
    print("• Appropriate tools for specific work")
    print("• Consistent output formats")
    print("• Scalable team-based analysis")


def demo_research_team_workflow():
    """Demonstrate the research team coordination pattern."""
    print("\n👥 RESEARCH TEAM WORKFLOW DEMO")
    print("=" * 50)

    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")):
        print("❌ API keys required for research team demo")
        return

    market_question = "Will there be a US recession in 2025?"

    print(f"\n🏗️ Research Team Analysis: {market_question}")

    print("\n👨‍💼 Research Team Members:")
    print("• Data Collector: Gathers raw information from multiple sources")
    print("• Quantitative Analyzer: Performs statistical analysis and modeling")
    print("• Synthesis Specialist: Integrates findings into final recommendations")

    print("\n🔄 Workflow:")
    print("1. Data Collector gathers comprehensive information")
    print("2. Quantitative Analyzer processes numbers and calculates probabilities")
    print("3. Synthesis Specialist combines everything into final assessment")

    try:
        result = research_team_analysis(market_question)

        print("
✅ Research Team Coordination Completed:"        print(f"   Response length: {len(result['analysis'])} characters")
        print("   Workflow: Collection → Analysis → Synthesis"
        print("\n📋 Team Benefits:")
        print("• Parallel processing of different aspects")
        print("• Specialized expertise for each phase")
        print("• Clean handoffs between team members")
        print("• Scalable analysis framework")

    except Exception as e:
        print(f"❌ Research team demo failed: {str(e)}")


def demo_context_bloat_prevention():
    """Demonstrate how subagents prevent context bloat."""
    print("\n💥 CONTEXT BLOAT PREVENTION DEMO")
    print("=" * 50)

    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")):
        print("❌ API keys required for context bloat demo")
        return

    market_question = "What are the latest developments in quantum computing?"

    print(f"\n🧠 Context Management Test: {market_question}")

    agent = create_polymarket_research_agent(
        storage_strategy="composite",
        enable_trading=False
    )

    print("\n📊 Context Bloat Scenario:")
    print("• Agent needs to research complex technical topic")
    print("• Multiple web searches return detailed technical content")
    print("• Without subagents: All search results go into main context")
    print("• With subagents: Detailed work isolated, main context stays clean")

    try:
        # Test with subagent isolation
        prompt = f"""
        Research this complex technical topic: {market_question}

        IMPORTANT: This is a context bloat test. Use subagents appropriately:
        1. For comprehensive research that would create many tool calls, use the general-purpose subagent
        2. Delegate detailed information gathering to specialized subagents
        3. Keep your main context clean by having subagents return summaries only
        4. Save detailed findings to files instead of returning raw data

        Demonstrate proper context management techniques.
        """

        result = agent.invoke({"messages": [{"role": "user", "content": prompt]})

        print("
✅ Context Management Test Completed:"        print(f"   Final response: {len(result['analysis'])} characters")
        print("   Technique: Subagent isolation + file-based storage"
        print("\n🧹 Context Management Benefits:")
        print("• Main agent context stays focused on coordination")
        print("• Detailed work isolated in subagent contexts")
        print("• Large data saved to files, not kept in memory")
        print("• Scalable to very complex research tasks")

    except Exception as e:
        print(f"❌ Context bloat prevention demo failed: {str(e)}")


def demo_subagent_best_practices():
    """Demonstrate best practices for subagent usage."""
    print("\n✅ SUBAGENT BEST PRACTICES DEMO")
    print("=" * 50)

    print("\n📚 Best Practices Implemented:")

    print("\n1️⃣ Clear, Specific Descriptions:")
    print("   ✓ 'Conducts comprehensive market research using web search and data analysis'")
    print("   ❌ 'Does research stuff'")

    print("\n2️⃣ Appropriate Tool Sets:")
    print("   ✓ market_researcher: web_search, market tools")
    print("   ✓ risk_analyzer: built-in tools only (calculations)")
    print("   ✓ trade_executor: trading tools only")

    print("\n3️⃣ Concise Result Formats:")
    print("   ✓ Structured output with word limits")
    print("   ✓ Summaries instead of raw data")
    print("   ✓ File-based storage for large content")

    print("\n4️⃣ Context Isolation:")
    print("   ✓ General-purpose subagent for complex work")
    print("   ✓ File system for intermediate results")
    print("   ✓ Clean handoffs between subagents")

    print("\n5️⃣ Specialized Roles:")
    print("   ✓ Different subagents for different expertise areas")
    print("   ✓ Appropriate models for specific tasks")
    print("   ✓ Focused system prompts")

    print("\n🎯 Result: Clean, scalable, and maintainable agent architecture")

    # Show actual subagent definitions
    agent = create_polymarket_research_agent(enable_trading=False)
    print("
🔍 Current Subagent Configuration:"    print(f"   Total subagents: {len(agent.subagents) if hasattr(agent, 'subagents') else 'N/A'}")
    print("   • Designed for context isolation and specialization")
    print("   • Optimized for prediction market analysis")
    print("   • Following deepagents best practices")


def main():
    """Run all subagent pattern demonstrations."""
    print("🎭 DeepAgents Subagent Patterns Demo")
    print("Advanced context isolation and specialized agent coordination")

    # Check environment
    has_keys = bool(os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY"))
    if not has_keys:
        print("\n⚠️  Note: Full demos require API keys")
        print("Set: ANTHROPIC_API_KEY and TAVILY_API_KEY")

    print("\n" + "=" * 60)

    # Run demos
    demo_context_isolation()
    demo_specialized_subagents()
    demo_research_team_workflow()
    demo_context_bloat_prevention()
    demo_subagent_best_practices()

    print("\n" + "=" * 60)
    print("✅ SUBAGENT PATTERNS DEMO COMPLETE")
    print("=" * 60)

    print("""
🎯 SUBAGENT PATTERNS SUMMARY:

🧹 Context Isolation:
• General-purpose subagent for complex work isolation
• File system for intermediate result storage
• Clean context management for long conversations

🎯 Specialization:
• market_researcher: Web search and data gathering
• risk_analyzer: Quantitative assessment and Kelly sizing
• data_synthesizer: Integration and final recommendations
• quick_researcher: Fast answers for simple questions

👥 Team Coordination:
• Data Collector → Quantitative Analyzer → Synthesis Specialist
• Parallel processing of different analysis aspects
• Clean handoffs and scalable workflows

📚 Best Practices:
• Clear, specific subagent descriptions
• Concise result formats (under 500 words)
• Appropriate tool sets for each role
• File-based storage to prevent context bloat

🚀 ADVANCED FEATURES:
• Automatic context isolation for complex tasks
• Specialized subagents for different expertise areas
• Research team coordination patterns
• Enterprise-grade context management

🎛️ RESULT: Clean, scalable agent architecture capable of
   complex multi-step analysis without context bloat!
""")


if __name__ == "__main__":
    main()

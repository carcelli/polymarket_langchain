#!/usr/bin/env python3
"""
Human-in-the-Loop (HITL) Demo for Polymarket Deep Agents

This script demonstrates advanced human approval workflows:
- Interrupt handling for sensitive operations
- Multiple tool call approvals
- Tool argument editing capabilities
- Risk-based decision controls
- Interactive trading sessions

Run with: python hitl_demo.py
"""

import os
import uuid
from agents.deep_research_agent import (
    create_trading_agent_with_approval,
    handle_agent_interrupt,
    create_human_decisions,
    resume_agent_with_decisions,
    interactive_trading_session
)

def demo_basic_interrupt_handling():
    """Demonstrate basic interrupt detection and handling."""
    print("🔄 BASIC INTERRUPT HANDLING DEMO")
    print("=" * 50)

    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")):
        print("❌ API keys required for HITL demos")
        return

    # Create agent with trading approval
    agent = create_trading_agent_with_approval()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    market_question = "Will Ethereum reach $10,000 by 2026?"

    print(f"📊 Testing interrupt handling for: {market_question}")

    try:
        # Trigger a trade proposal that requires approval
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Analyze this market and if you find a good opportunity, execute a trade: {market_question}"
            }]
        }, config=config)

        # Check for interrupts
        needs_approval, interrupt_info = handle_agent_interrupt(result, config)

        if needs_approval:
            print("✅ Interrupt detected!")
            print(f"Thread ID: {interrupt_info['thread_id']}")

            action_requests = interrupt_info["action_requests"]
            print(f"Pending actions: {len(action_requests)}")

            for i, action in enumerate(action_requests, 1):
                print(f"{i}. {action['name']}: {action['args']}")

        else:
            print("ℹ️  No interrupts - agent completed without requiring approval")
            print(f"Result: {result['messages'][-1].content[:200]}...")

    except Exception as e:
        print(f"❌ Basic interrupt demo failed: {str(e)}")


def demo_multiple_tool_approvals():
    """Demonstrate handling multiple tool calls requiring approval."""
    print("\n📋 MULTIPLE TOOL APPROVALS DEMO")
    print("=" * 50)

    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")):
        print("❌ API keys required for multiple approvals demo")
        return

    agent = create_trading_agent_with_approval()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("🎯 Testing multiple simultaneous approvals")
    print("Agent will attempt multiple trading operations...")

    try:
        # Trigger multiple trade operations
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Execute multiple trades: a market buy for YES and a limit sell for NO on Bitcoin market"
            }]
        }, config=config)

        needs_approval, interrupt_info = handle_agent_interrupt(result, config)

        if needs_approval:
            action_requests = interrupt_info["action_requests"]
            review_configs = interrupt_info["review_configs"]

            print(f"✅ {len(action_requests)} actions require approval:")
            print()

            # Display all pending actions
            for i, action in enumerate(action_requests, 1):
                config = review_configs[action["name"]]
                print(f"{i}. 🔧 {action['name']}")
                print(f"   📝 Args: {action['args']}")
                print(f"   ✅ Allowed: {config['allowed_decisions']}")
                print()

            # Simulate user decisions for each action
            print("🤖 Simulated user decisions:")
            user_decisions = [
                {"type": "approve"} if i % 2 == 0 else {"type": "reject"}
                for i in range(len(action_requests))
            ]

            for i, decision in enumerate(user_decisions, 1):
                action_name = action_requests[i-1]["name"]
                print(f"   {i}. {action_name}: {decision['type']}")

            # Create properly formatted decisions
            decisions = create_human_decisions(action_requests, review_configs, user_decisions)
            print(f"\n📤 Formatted decisions: {decisions}")

            # Resume execution
            final_result = resume_agent_with_decisions(agent, decisions, config)
            print("\n✅ Multiple approvals processed successfully!")

        else:
            print("ℹ️  No multiple approvals needed")

    except Exception as e:
        print(f"❌ Multiple approvals demo failed: {str(e)}")


def demo_tool_argument_editing():
    """Demonstrate editing tool arguments before approval."""
    print("\n✏️  TOOL ARGUMENT EDITING DEMO")
    print("=" * 50)

    if not (os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY")):
        print("❌ API keys required for editing demo")
        return

    agent = create_trading_agent_with_approval()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("🎯 Testing tool argument editing capabilities")
    print("Agent will propose a trade, then we'll modify the parameters...")

    try:
        # Get agent to propose a trade
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": "Propose a market buy order for YES on a crypto market"
            }]
        }, config=config)

        needs_approval, interrupt_info = handle_agent_interrupt(result, config)

        if needs_approval:
            action_requests = interrupt_info["action_requests"]

            if action_requests:
                action = action_requests[0]
                print(f"📝 Original action: {action['name']}")
                print(f"   Original args: {action['args']}")
                print()

                # Simulate user editing the trade parameters
                print("✏️  User edits the trade parameters:")
                edited_action = {
                    "name": action["name"],
                    "args": {
                        "token_id": "modified_token_id",
                        "amount": 500.0,  # Modified amount
                        "side": "BUY"
                    }
                }

                user_decisions = [{
                    "type": "edit",
                    "edited_action": edited_action
                }]

                print(f"   ✏️  Modified amount: {edited_action['args']['amount']}")
                print(f"   🆔 Modified token: {edited_action['args']['token_id']}")

                # Create and apply decisions
                review_configs = interrupt_info["review_configs"]
                decisions = create_human_decisions(action_requests, review_configs, user_decisions)

                final_result = resume_agent_with_decisions(agent, decisions, config)
                print("\n✅ Trade executed with edited parameters!")

            else:
                print("ℹ️  No editable actions proposed")

        else:
            print("ℹ️  No editing opportunities")

    except Exception as e:
        print(f"❌ Editing demo failed: {str(e)}")


def demo_risk_based_configuration():
    """Demonstrate different interrupt configurations for different risk levels."""
    print("\n⚠️  RISK-BASED CONFIGURATION DEMO")
    print("=" * 50)

    print("🎛️  Risk-Based Interrupt Strategies:")
    print()

    risk_configs = {
        "Conservative (High Safety)": {
            "trading": {"allowed_decisions": ["approve", "reject"]},  # No editing
            "research": {"allowed_decisions": ["approve", "reject"]},  # API calls need approval
            "description": "Maximum human oversight, minimal automation"
        },
        "Moderate (Balanced)": {
            "trading": {"allowed_decisions": ["approve", "edit", "reject"]},  # Full control
            "research": False,  # No interrupts for research
            "description": "Human oversight for trades, automation for research"
        },
        "Aggressive (High Automation)": {
            "trading": False,  # No interrupts
            "research": False,  # Full automation
            "description": "Minimal human intervention, high automation"
        }
    }

    for risk_level, config in risk_configs.items():
        print(f"🛡️  {risk_level}:")
        print(f"   {config['description']}")
        print(f"   Trading interrupts: {config['trading']}")
        print(f"   Research interrupts: {config['research']}")
        print()

    print("💡 Configuration Guidelines:")
    print("• High-risk operations: approve/edit/reject (full control)")
    print("• Medium-risk operations: approve/reject (no editing)")
    print("• Low-risk operations: False (no interrupts)")
    print("• Scale based on deployment environment and trust levels")


def demo_subagent_interrupts():
    """Demonstrate subagent-specific interrupt configurations."""
    print("\n🎭 SUBAGENT INTERRUPT CONFIGURATION DEMO")
    print("=" * 50)

    print("👥 Subagent Interrupt Capabilities:")
    print()
    print("• Main agent can have different interrupt policies")
    print("• Subagents can override main agent settings")
    print("• trade_executor subagent requires approval for all trades")
    print("• market_researcher operates autonomously")
    print("• risk_analyzer focuses on calculations only")
    print()

    print("🔧 Configuration Example:")
    print("""
subagents = [
    {
        "name": "trade_executor",
        "interrupt_on": {
            "execute_market_order": {"allowed_decisions": ["approve", "edit", "reject"]},
            "execute_limit_order": {"allowed_decisions": ["approve", "reject"]},
        }
    }
]
""")

    print("🎯 Benefits:")
    print("• Granular control over different agent components")
    print("• Specialized safety policies for different operations")
    print("• Override main agent settings for specific subagents")


def demo_complete_workflow():
    """Demonstrate the complete human-in-the-loop workflow."""
    print("\n🚀 COMPLETE HITL WORKFLOW DEMO")
    print("=" * 50)

    print("🔄 Full Human-in-the-Loop Workflow:")
    print("1. Agent analyzes market and proposes actions")
    print("2. Sensitive operations trigger interrupts")
    print("3. Human reviews pending actions")
    print("4. Human makes approve/edit/reject decisions")
    print("5. Agent resumes with human decisions")
    print("6. Approved actions execute, rejected actions skip")
    print()

    # Run the interactive trading session demo
    print("🎮 Running Interactive Trading Session...")
    print("-" * 40)
    interactive_trading_session()


def main():
    """Run all human-in-the-loop demonstrations."""
    print("👥 Human-in-the-Loop (HITL) Demo for Polymarket Deep Agents")
    print("Advanced interrupt handling and human approval workflows")

    # Check environment
    has_keys = bool(os.getenv("ANTHROPIC_API_KEY") and os.getenv("TAVILY_API_KEY"))
    if not has_keys:
        print("\n⚠️  Note: Full demos require API keys")
        print("Set: ANTHROPIC_API_KEY and TAVILY_API_KEY")

    print("\n" + "=" * 60)

    # Run demos
    demo_basic_interrupt_handling()
    demo_multiple_tool_approvals()
    demo_tool_argument_editing()
    demo_risk_based_configuration()
    demo_subagent_interrupts()
    demo_complete_workflow()

    print("\n" + "=" * 60)
    print("✅ HUMAN-IN-THE-LOOP DEMO COMPLETE")
    print("=" * 60)

    print("""
🎯 HITL CAPABILITIES SUMMARY:

🔄 Interrupt Handling:
• Automatic pause on sensitive operations
• State persistence across approval cycles
• Resume execution with human decisions

📋 Multiple Approvals:
• Batch processing of multiple tool calls
• Ordered decision handling
• Consistent approval workflows

✏️ Argument Editing:
• Modify tool parameters before execution
• Full control over execution details
• Validation and safety checks

⚠️ Risk-Based Configuration:
• Conservative: approve/reject only (no editing)
• Moderate: approve/edit/reject (full control)
• Aggressive: no interrupts (full automation)

👥 Subagent Interrupts:
• Per-subagent interrupt policies
• Override main agent settings
• Specialized safety controls

🚀 Production Features:
• Checkpointer integration for state persistence
• Thread-safe operation with unique IDs
• Enterprise-grade approval workflows

🎛️ RESULT: Sophisticated human oversight capabilities
   enabling safe, controlled AI agent operation!
""")


if __name__ == "__main__":
    main()

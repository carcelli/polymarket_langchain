"""
Command Pattern - Fluent Python Chapter 6 Style

Demonstrates how the Command pattern can be simplified using functions
instead of classes with single methods (pages 177-178, 323).

Traditional Command Pattern (verbose):
    class ConcreteCommand:
        def __init__(self, receiver):
            self.receiver = receiver
        def execute(self):
            self.receiver.action()

Fluent Python Approach (simple):
    def command_function(receiver):
        receiver.action()

Benefits (p. 178):
- Functions are lightweight compared to single-method classes
- No need for Flyweight pattern since functions are naturally shared
- Functions can be combined in MacroCommands using __call__
- Easier to test and compose

This implementation shows how market operations can be modeled
as simple functions that become reusable commands.
"""

from typing import Callable, List, Dict, Any
from .registry import register_strategy


# Command functions - simple functions instead of command classes
def analyze_market_command(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Command to analyze a market."""
    return {
        "action": "analyze",
        "market_id": market_data.get("id"),
        "analysis": f"Analyzed market {market_data.get('question', 'Unknown')}",
    }


def place_bet_command(
    market_data: Dict[str, Any], amount: float = 100
) -> Dict[str, Any]:
    """Command to place a bet."""
    return {
        "action": "bet",
        "market_id": market_data.get("id"),
        "amount": amount,
        "message": f"Placed ${amount} bet on market {market_data.get('question', 'Unknown')}",
    }


def monitor_market_command(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """Command to start monitoring a market."""
    return {
        "action": "monitor",
        "market_id": market_data.get("id"),
        "message": f"Started monitoring market {market_data.get('question', 'Unknown')}",
    }


class MacroCommand:
    """
    MacroCommand using functions as commands.

    From Fluent Python: "Instead of giving the Invoker a Command instance,
    we can simply give it a function... The MacroCommand can be implemented
    with a class implementing __call__."

    This is simpler than the traditional Command pattern with separate classes
    for each command type.
    """

    def __init__(self, commands: List[Callable]):
        self.commands = list(commands)  # Store copies

    def __call__(self, *args, **kwargs):
        """Execute all commands in sequence."""
        results = []
        for command in self.commands:
            try:
                result = command(*args, **kwargs)
                results.append(result)
            except Exception as e:
                results.append({"error": str(e), "command": command.__name__})
        return results


# Pre-configured command sequences
@register_strategy("market_ops_suite")
def market_operations_suite(market_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Meta-command that runs a suite of market operations.

    Demonstrates the Flyweight pattern benefit: functions are shared objects
    that can be used in multiple contexts simultaneously (page 196).
    """
    # Create macro command with function-based commands
    macro = MacroCommand(
        [
            analyze_market_command,
            lambda data: place_bet_command(data, 50),  # Partial application
            monitor_market_command,
        ]
    )

    # Execute the command suite
    results = macro(market_data)

    return {
        "strategy": "market_operations_suite",
        "results": results,
        "summary": f"Executed {len(results)} operations on market {market_data.get('id')}",
    }


# Example usage demonstrating function-based commands
if __name__ == "__main__":
    market_data = {"id": "test_market_123", "question": "Will it rain tomorrow?"}

    # Direct command usage
    print("🔧 Direct command execution:")
    print(analyze_market_command(market_data))
    print(place_bet_command(market_data, 75))

    # Macro command usage
    print("\n📋 Macro command execution:")
    macro = MacroCommand([analyze_market_command, monitor_market_command])
    results = macro(market_data)
    for result in results:
        print(f"  • {result}")

    # Strategy-based command suite
    print("\n🎯 Strategy-based command suite:")
    from .registry import best_strategy

    result = best_strategy(market_data)
    print(f"Strategy result: {result.get('summary', 'No summary')}")

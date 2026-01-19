"""
ML Strategy Registry - Fluent Python Chapters 6-7 Style

Registration and discovery system for ML betting strategies.
Implements the Strategy pattern using first-class functions (Chapter 6)
and function decorators (Chapter 7).

Key Concepts Implemented:
- Strategy Pattern: Functions instead of classes for strategies (p. 174-175)
- Registration Decorator: Avoids manual registry maintenance (p. 187)
- Meta-Strategy: best_strategy() selects highest-edge strategy (p. 175)
- Error Isolation: Failed strategies don't break the system (p. 175)
- Functions as Flyweights: Shared, reusable strategy objects (p. 196)

Traditional approach (manual lists):
    promos = [fidelity_promo, bulk_item_promo, large_order_promo]

Our approach (automatic registration):
    @register_strategy
    def my_strategy(market_data): ...

Benefits:
- Zero manual registry maintenance
- Automatic discovery via introspection
- Error isolation between strategies
- Support for both function and class-based strategies
"""

from typing import Callable, List, Dict, Any, Optional
import inspect
import importlib
import pkgutil
import sys
from pathlib import Path

# Global registry: strategy_name → strategy_callable
STRATEGIES: Dict[str, Callable] = {}

def register_strategy(name: str | None = None):
    """
    Decorator to register a strategy function or class.

    Usage:
        @register_strategy("my_strategy")
        def my_function_strategy(market_data):
            return {"edge": 0.05, "recommendation": "BUY"}

        @register_strategy("xgboost_v1")
        class XGBoostStrategy(MLBettingStrategy):
            def predict(self, market_data):
                # ... implementation
                pass
    """
    def decorator(strategy_obj):
        strategy_name = name or strategy_obj.__name__
        STRATEGIES[strategy_name] = strategy_obj
        print(f"✅ Registered strategy: {strategy_name}")
        return strategy_obj
    return decorator

def discover_strategies(module_path: str = "polymarket_agents.ml_strategies"):
    """
    Auto-discover strategies via module introspection.
    Looks for callables with 'predict' method or registered strategies.
    """
    try:
        package = importlib.import_module(module_path)

        # Get the package path
        if hasattr(package, '__path__'):
            for _, module_name, _ in pkgutil.iter_modules(package.__path__):
                try:
                    module = importlib.import_module(f"{module_path}.{module_name}")

                    # Look for strategy objects
                    for name, obj in inspect.getmembers(module):
                        # Skip private items and already registered items
                        if name.startswith('_') or name in STRATEGIES:
                            continue

                        # Check if it's a strategy (has predict method or is callable)
                        if (inspect.isclass(obj) and hasattr(obj, 'predict')) or \
                           (inspect.isfunction(obj) and 'predict' in name):
                            STRATEGIES[name] = obj
                            print(f"🔍 Auto-discovered strategy: {name}")

                except Exception as e:
                    print(f"⚠️ Failed to load module {module_name}: {e}")

    except Exception as e:
        print(f"⚠️ Failed to discover strategies: {e}")

def get_available_strategies() -> List[str]:
    """Return list of available strategy names."""
    return list(STRATEGIES.keys())

def get_strategy(name: str) -> Optional[Callable]:
    """Get a strategy by name."""
    return STRATEGIES.get(name)

def best_strategy(market_data: Dict[str, Any], min_edge: float = 0.0) -> Dict[str, Any]:
    """
    Meta-strategy: run all registered strategies and pick the one with highest edge.

    Returns the recommendation from the winning strategy, or error if none qualify.
    """
    results = []

    for name, strategy in STRATEGIES.items():
        try:
            # Support both function and class interfaces
            if inspect.isclass(strategy):
                # Instantiate and call predict
                predictor = strategy()
                if hasattr(predictor, 'predict'):
                    pred = predictor.predict(market_data)
                else:
                    continue  # Not a valid strategy
            elif callable(strategy):
                # Direct function call
                pred = strategy(market_data)
            else:
                continue  # Not callable

            # Normalize prediction format
            if isinstance(pred, dict):
                edge = pred.get("edge", 0.0)
                if edge >= min_edge:
                    results.append((edge, name, pred))
            elif hasattr(pred, 'edge'):
                # StrategyResult object
                edge = pred.edge
                if edge >= min_edge:
                    results.append((edge, name, pred))

        except Exception as e:
            print(f"❌ Strategy {name} failed: {e}")
            continue

    if not results:
        return {"error": "No qualifying strategies available", "strategies_tried": len(STRATEGIES)}

    # Sort by edge descending and pick best
    results.sort(key=lambda x: x[0], reverse=True)
    best_edge, best_name, best_pred = results[0]

    # Add metadata
    if isinstance(best_pred, dict):
        best_pred.update({
            "selected_strategy": best_name,
            "edge": best_edge,
            "strategies_compared": len(results),
            "total_strategies": len(STRATEGIES)
        })
    else:
        # For StrategyResult objects, we can't modify directly
        result_dict = {
            "selected_strategy": best_name,
            "edge": best_edge,
            "strategies_compared": len(results),
            "total_strategies": len(STRATEGIES),
            "prediction": best_pred
        }
        return result_dict

    return best_pred

# Auto-discover strategies on module import
print("🔧 Initializing ML Strategy Registry...")
discover_strategies()
print(f"📊 Available strategies: {get_available_strategies()}")
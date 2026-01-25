#!/usr/bin/env python3
"""
Test script for Polymarket LangChain agents.

This demonstrates the key agent functionalities:
1. Probability extraction from Polymarket data
2. ML forecast comparison against market consensus
3. Business domain-specific analysis

Usage:
    python test_polymarket_agents.py

Requirements:
    - OPENAI_API_KEY environment variable set
    - Package installed: pip install -e .
"""

import os
from dotenv import load_dotenv
from polymarket_agents.langchain.agent import (
    create_probability_extraction_agent,
    compare_ml_vs_market_forecast,
    create_crypto_agent,
    create_sports_agent,
    analyze_business_risks,
    run_agent,
)

load_dotenv()


def test_probability_extraction():
    """Test the probability extraction agent."""
    print("=" * 60)
    print("TESTING: Probability Extraction Agent")
    print("=" * 60)

    try:
        agent = create_probability_extraction_agent()

        query = """
        Extract implied probabilities from current Polymarket data for economic events.
        Focus on US recession risks, Fed rate decisions, and GDP forecasts.
        Return the top 5 markets with highest volume and their implied probabilities.
        """

        print("Query:", query.strip())
        print("\nRunning agent...")
        result = run_agent(agent, query)

        print("\nResult:")
        print(result[:500] + "..." if len(result) > 500 else result)
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_ml_comparison():
    """Test ML forecast comparison."""
    print("\n" + "=" * 60)
    print("TESTING: ML Forecast Comparison")
    print("=" * 60)

    try:
        # Example: Your ML model predicts 35% recession probability
        ml_forecast = 0.35
        event = "US recession in 2026"

        print(f"Your ML Forecast: {ml_forecast:.1%} probability of {event}")

        comparison = compare_ml_vs_market_forecast(
            ml_forecast=ml_forecast, event_description=event
        )

        print("\nComparison Analysis:")
        result = comparison["comparison"]
        print(result[:500] + "..." if len(result) > 500 else result)
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_crypto_agent():
    """Test crypto-focused agent."""
    print("\n" + "=" * 60)
    print("TESTING: Crypto Agent")
    print("=" * 60)

    try:
        agent = create_crypto_agent()

        query = """
        Analyze current Polymarket data for crypto markets.
        Focus on Bitcoin and Ethereum price predictions, adoption rates, and regulation.
        What are the key crypto market opportunities and risks?
        """

        print("Query:", query.strip())
        result = run_agent(agent, query)

        print("\nResult:")
        print(result[:500] + "..." if len(result) > 500 else result)
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_business_risks():
    """Test business risk analysis function (crypto-focused)."""
    print("\n" + "=" * 60)
    print("TESTING: Crypto Business Risk Analysis")
    print("=" * 60)

    try:
        business_type = "crypto trading and investment firm"
        domain = "crypto"

        print(f"Business Type: {business_type}")
        print(f"Risk Domain: {domain}")

        analysis = analyze_business_risks(business_type=business_type, domain=domain)

        print("\nRisk Analysis:")
        print(analysis[:500] + "..." if len(analysis) > 500 else analysis)
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_sports_agent():
    """Test sports-focused agent."""
    print("\n" + "=" * 60)
    print("TESTING: Sports Agent")
    print("=" * 60)

    try:
        agent = create_sports_agent()

        query = """
        Analyze current Polymarket data for sports markets.
        Focus on major championships, tournament outcomes, and betting opportunities.
        What are the most interesting sports markets to watch?
        """

        print("Query:", query.strip())
        result = run_agent(agent, query)

        print("\nResult:")
        print(result[:500] + "..." if len(result) > 500 else result)
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all tests."""
    print("Polymarket LangChain Agents - Test Suite")
    print("=========================================")

    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key before running tests.")
        return

    tests = [
        ("Probability Extraction", test_probability_extraction),
        ("ML Comparison", test_ml_comparison),
        ("Crypto Agent", test_crypto_agent),
        ("Sports Agent", test_sports_agent),
        ("Business Risk Analysis", test_business_risks),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        success = test_func()
        results.append((test_name, success))
        print(f"{'✓ PASSED' if success else '✗ FAILED'}: {test_name}")

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Your Polymarket agents are ready to use.")
        print("\nNext steps:")
        print("1. Test with real market data")
        print("2. Customize agent prompts for your business")
        print("3. Integrate with your ML forecasting pipeline")
        print("4. Add monitoring and alerting for key markets")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Check the errors above.")


if __name__ == "__main__":
    main()

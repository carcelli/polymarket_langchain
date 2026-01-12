#!/usr/bin/env python3
"""
Real-World Market Analysis Workflow

This script demonstrates how to use LangGraph agents with real Polymarket data
for systematic market analysis and opportunity identification.
"""

import sys
import os
import sqlite3
import json
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parents[2] / "src"))

from polymarket_agents.graph.planning_agent import create_planning_agent, analyze_bet
from polymarket_agents.graph.memory_agent import create_memory_agent


class MarketAnalyzer:
    """Real-world market analysis workflow using LangGraph agents."""

    def __init__(self, db_path: str = "data/markets.db"):
        self.db_path = db_path
        self.planning_agent = create_planning_agent()
        self.memory_agent = create_memory_agent()

    def get_high_volume_markets(self, category: str = None, limit: int = 10, min_volume: float = 1000000) -> List[Dict]:
        """Get high-volume markets from local database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = '''
            SELECT id, question, category, volume, outcome_prices, active, end_date
            FROM markets
            WHERE active = 1 AND volume > ?
        '''

        params = [min_volume]

        if category:
            query += ' AND category = ?'
            params.append(category.lower())

        query += ' ORDER BY volume DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        markets = cursor.fetchall()
        conn.close()

        market_data = []
        for market in markets:
            market_id, question, category, volume, prices, active, end_date = market
            try:
                price_list = eval(prices) if prices else ['0', '0']
                yes_price = float(price_list[0]) if len(price_list) > 0 else 0
                no_price = float(price_list[1]) if len(price_list) > 1 else 0
                implied_prob = yes_price * 100
            except:
                implied_prob = 0

            market_data.append({
                'id': market_id,
                'question': question,
                'category': category,
                'volume': volume,
                'implied_probability': implied_prob,
                'end_date': end_date,
                'active': bool(active)
            })

        return market_data

    def analyze_market_opportunity(self, market_question: str) -> Dict[str, Any]:
        """Run full analysis on a specific market using planning agent."""
        print(f"\\n🔍 Analyzing: {market_question}")

        try:
            # Run planning agent analysis
            result = analyze_bet(market_question)

            if 'error' in result and result['error']:
                return {
                    'question': market_question,
                    'error': result['error'],
                    'analyzed_at': datetime.now().isoformat()
                }

            # Extract recommendation
            rec = result.get('recommendation', {})

            analysis = {
                'question': rec.get('market_question', market_question),
                'action': rec.get('action', 'UNKNOWN'),
                'recommended_side': rec.get('recommended_side', 'UNKNOWN'),
                'edge': rec.get('edge', 0),
                'expected_value': rec.get('expected_value', 0),
                'kelly_fraction': rec.get('kelly_fraction', 0),
                'analyzed_at': datetime.now().isoformat(),
                'full_result': result
            }

            return analysis

        except Exception as e:
            return {
                'question': market_question,
                'error': str(e),
                'analyzed_at': datetime.now().isoformat()
            }

    def find_opportunities_by_category(self, category: str, min_volume: float = 5000000) -> List[Dict]:
        """Find and analyze opportunities in a specific category."""
        print(f"\\n🎯 Scanning {category.upper()} markets (min volume: ${min_volume:,.0f})")

        # Get markets in category
        markets = self.get_high_volume_markets(category=category, min_volume=min_volume)

        if not markets:
            print(f"❌ No {category} markets found above volume threshold")
            return []

        print(f"📊 Found {len(markets)} high-volume {category} markets")

        # Analyze each market
        opportunities = []
        for market in markets:
            analysis = self.analyze_market_opportunity(market['question'])

            # Add market metadata
            analysis.update({
                'market_id': market['id'],
                'volume': market['volume'],
                'implied_probability': market['implied_probability'],
                'category': category
            })

            opportunities.append(analysis)

        return opportunities

    def generate_opportunity_report(self, opportunities: List[Dict]) -> str:
        """Generate a formatted report of opportunities."""
        report = []
        report.append("📊 MARKET OPPORTUNITY REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Filter for actionable opportunities
        actionable = [opp for opp in opportunities if opp.get('action') in ['BUY', 'SELL'] and not opp.get('error')]

        if not actionable:
            report.append("❌ No actionable opportunities found")
            report.append("")
            report.append("Markets analyzed:")
            for opp in opportunities[:5]:  # Show first 5
                status = "❌ Error" if opp.get('error') else f"⏭️ {opp.get('action', 'UNKNOWN')}"
                report.append(f"• {opp['question'][:60]}... - {status}")
            return "\\n".join(report)

        # Sort by edge (best opportunities first)
        actionable.sort(key=lambda x: abs(x.get('edge', 0)), reverse=True)

        report.append(f"🎯 ACTIONABLE OPPORTUNITIES: {len(actionable)}")
        report.append("")

        for i, opp in enumerate(actionable[:10], 1):  # Top 10
            report.append(f"{i}. {opp['question']}")
            report.append(f"   📊 Volume: ${opp['volume']:,.0f} | Implied Prob: {opp['implied_probability']:.1f}%")
            report.append(f"   🎯 Action: {opp['action']} {opp['recommended_side']}")
            report.append(f"   📈 Edge: {opp['edge']:.2f}% | EV: {opp['expected_value']:.3f}")
            report.append(f"   💰 Kelly: {opp['kelly_fraction']:.2f}%")
            report.append("")

        return "\\n".join(report)

    def run_comprehensive_scan(self, categories: List[str] = None) -> str:
        """Run comprehensive opportunity scan across categories."""
        if categories is None:
            categories = ['politics', 'sports', 'crypto', 'tech', 'geopolitics']

        all_opportunities = []

        for category in categories:
            try:
                opportunities = self.find_opportunities_by_category(category)
                all_opportunities.extend(opportunities)
                print(f"✅ Completed {category} analysis")
            except Exception as e:
                print(f"❌ Error analyzing {category}: {e}")

        # Generate final report
        report = self.generate_opportunity_report(all_opportunities)

        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_analysis_report_{timestamp}.txt"

        with open(filename, 'w') as f:
            f.write(report)

        print(f"\\n📄 Report saved to: {filename}")
        return report


def main():
    """Main demonstration workflow."""
    print("🚀 Polymarket LangGraph Analysis Workflow")
    print("=" * 50)

    analyzer = MarketAnalyzer()

    # Example 1: Analyze a specific high-volume market
    print("\\n" + "="*50)
    print("EXAMPLE 1: Specific Market Analysis")
    print("="*50)

    specific_market = "Russia x Ukraine ceasefire in 2025?"
    analysis = analyzer.analyze_market_opportunity(specific_market)
    print(f"Result: {analysis.get('action', 'UNKNOWN')} - Edge: {analysis.get('edge', 0):.2f}%")

    # Example 2: Category scan
    print("\\n" + "="*50)
    print("EXAMPLE 2: Category Opportunity Scan")
    print("="*50)

    geopolitics_opportunities = analyzer.find_opportunities_by_category('geopolitics', min_volume=10000000)

    if geopolitics_opportunities:
        actionable = [opp for opp in geopolitics_opportunities if opp.get('action') in ['BUY', 'SELL']]
        print(f"Found {len(actionable)} actionable geopolitics opportunities")

    # Example 3: Comprehensive scan (commented out for demo)
    print("\\n" + "="*50)
    print("EXAMPLE 3: Full Market Scan (Demo Mode)")
    print("="*50)
    print("💡 To run full scan: analyzer.run_comprehensive_scan()")
    print("   This will analyze high-volume markets across all categories")
    print("   and generate a comprehensive opportunity report")


if __name__ == "__main__":
    main()

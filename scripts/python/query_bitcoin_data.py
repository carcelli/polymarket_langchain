"""
Query Bitcoin Market Data for ML Training

Utility script to extract and analyze collected Bitcoin market data.
Provides ML-ready datasets with features and labels.

Usage:
    python scripts/python/query_bitcoin_data.py --stats
    python scripts/python/query_bitcoin_data.py --export csv
    python scripts/python/query_bitcoin_data.py --market 574073
    python scripts/python/query_bitcoin_data.py --ml-ready --min-quality 0.7
"""

import argparse
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta


DB_PATH = Path("data/bitcoin_tracker.db")


def get_stats(db_path: Path):
    """Print database statistics."""
    conn = sqlite3.connect(str(db_path))

    print("\n" + "=" * 70)
    print("📊 BITCOIN MARKET TRACKER - DATABASE STATISTICS")
    print("=" * 70)

    # Total snapshots
    total = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM market_snapshots", conn
    ).iloc[0]["count"]
    print(f"\n📸 Total Snapshots: {total:,}")

    # Unique markets
    markets = pd.read_sql_query(
        "SELECT COUNT(DISTINCT market_id) as count FROM market_snapshots", conn
    ).iloc[0]["count"]
    print(f"🎯 Unique Markets: {markets}")

    # Date range
    dates = pd.read_sql_query(
        """
        SELECT 
            MIN(timestamp) as first,
            MAX(timestamp) as last
        FROM market_snapshots
    """,
        conn,
    )
    print(
        f"📅 Date Range: {dates.iloc[0]['first'][:10]} to {dates.iloc[0]['last'][:10]}"
    )

    # Resolved vs unresolved
    resolved_stats = pd.read_sql_query(
        """
        SELECT resolved, COUNT(*) as count
        FROM market_snapshots
        GROUP BY resolved
    """,
        conn,
    )
    print(
        f"\n✅ Resolved Markets: {resolved_stats[resolved_stats['resolved'] == 1]['count'].values[0] if 1 in resolved_stats['resolved'].values else 0:,}"
    )
    print(
        f"⏳ Unresolved Markets: {resolved_stats[resolved_stats['resolved'] == 0]['count'].values[0] if 0 in resolved_stats['resolved'].values else 0:,}"
    )

    # Top markets by snapshots
    print(f"\n🔥 Most Tracked Markets:")
    top_markets = pd.read_sql_query(
        """
        SELECT 
            question,
            COUNT(*) as snapshots,
            MIN(yes_price) as min_price,
            MAX(yes_price) as max_price,
            AVG(volume) as avg_volume
        FROM market_snapshots
        GROUP BY market_id
        ORDER BY snapshots DESC
        LIMIT 5
    """,
        conn,
    )

    for i, row in top_markets.iterrows():
        print(f"  {i+1}. {row['question'][:60]}...")
        print(
            f"     Snapshots: {row['snapshots']}, Price Range: {row['min_price']:.3f}-{row['max_price']:.3f}"
        )

    # Data quality distribution
    quality = pd.read_sql_query(
        """
        SELECT 
            CASE 
                WHEN data_quality_score >= 0.8 THEN 'High (≥0.8)'
                WHEN data_quality_score >= 0.5 THEN 'Medium (0.5-0.8)'
                ELSE 'Low (<0.5)'
            END as quality_tier,
            COUNT(*) as count
        FROM market_snapshots
        GROUP BY quality_tier
    """,
        conn,
    )

    print(f"\n📈 Data Quality Distribution:")
    for _, row in quality.iterrows():
        print(f"  {row['quality_tier']}: {row['count']:,} snapshots")

    # Recent collection activity
    recent = pd.read_sql_query(
        """
        SELECT COUNT(*) as count
        FROM market_snapshots
        WHERE datetime(timestamp) > datetime('now', '-24 hours')
    """,
        conn,
    )
    print(f"\n⏰ Last 24 Hours: {recent.iloc[0]['count']} snapshots")

    conn.close()
    print("\n" + "=" * 70 + "\n")


def export_data(db_path: Path, format: str = "csv", output_path: str = None):
    """Export data to CSV or JSON."""
    conn = sqlite3.connect(str(db_path))

    df = pd.read_sql_query(
        """
        SELECT 
            timestamp,
            market_id,
            question,
            yes_price,
            no_price,
            implied_probability,
            volume,
            liquidity,
            btc_spot_price,
            btc_24h_change_pct,
            price_momentum_15m,
            price_momentum_1h,
            volume_spike,
            price_volatility,
            rsi_14,
            market_edge,
            time_to_expiry_hours,
            resolved,
            outcome,
            data_quality_score
        FROM market_snapshots
        ORDER BY timestamp DESC
    """,
        conn,
    )

    if output_path is None:
        output_path = (
            f"bitcoin_market_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        )

    if format == "csv":
        df.to_csv(output_path, index=False)
    elif format == "json":
        df.to_json(output_path, orient="records", indent=2)

    print(f"✅ Exported {len(df)} records to {output_path}")
    conn.close()


def get_market_history(db_path: Path, market_id: str):
    """Get time series for a specific market."""
    conn = sqlite3.connect(str(db_path))

    df = pd.read_sql_query(
        """
        SELECT *
        FROM market_snapshots
        WHERE market_id = ?
        ORDER BY timestamp
    """,
        conn,
        params=(market_id,),
    )

    if df.empty:
        print(f"❌ No data found for market {market_id}")
        return

    print(f"\n📊 Market: {df.iloc[0]['question']}")
    print(f"📸 Snapshots: {len(df)}")
    print(
        f"📅 Period: {df.iloc[0]['timestamp'][:10]} to {df.iloc[-1]['timestamp'][:10]}"
    )
    print(f"💵 Price Range: {df['yes_price'].min():.3f} - {df['yes_price'].max():.3f}")
    print(f"📈 Current Price: {df.iloc[-1]['yes_price']:.3f}")

    if df.iloc[-1]["resolved"]:
        print(f"✅ Resolved: {df.iloc[-1]['outcome']}")

    # Show price trajectory
    print(f"\n📉 Recent Price History:")
    for i in range(min(10, len(df))):
        row = df.iloc[-(i + 1)]
        print(
            f"  {row['timestamp'][:19]} | Price: {row['yes_price']:.3f} | BTC: ${row['btc_spot_price']:,.0f}"
        )

    conn.close()


def get_ml_ready_dataset(
    db_path: Path,
    min_quality: float = 0.5,
    include_unresolved: bool = False,
    output_path: str = None,
) -> pd.DataFrame:
    """
    Export ML-ready dataset with features and labels.

    Args:
        min_quality: Minimum data quality score (0-1)
        include_unresolved: Include markets without outcomes (for prediction)
        output_path: Optional path to save dataset

    Returns:
        DataFrame ready for sklearn/training
    """
    conn = sqlite3.connect(str(db_path))

    # Build query
    where_clause = f"data_quality_score >= {min_quality}"
    if not include_unresolved:
        where_clause += " AND resolved = 1"

    query = f"""
        SELECT 
            -- Features
            yes_price as market_probability,
            volume,
            liquidity,
            btc_spot_price,
            btc_24h_change_pct,
            price_momentum_15m,
            price_momentum_1h,
            volume_spike,
            price_volatility,
            rsi_14,
            market_edge,
            time_to_expiry_hours,
            data_quality_score,
            
            -- Labels (for supervised learning)
            CASE WHEN outcome = 'YES' THEN 1 ELSE 0 END as label_yes,
            resolved,
            
            -- Metadata
            market_id,
            timestamp,
            question
        FROM market_snapshots
        WHERE {where_clause}
        ORDER BY timestamp DESC
    """

    df = pd.read_sql_query(query, conn)

    # Drop rows with missing critical features
    feature_cols = [
        "market_probability",
        "volume",
        "liquidity",
        "price_momentum_15m",
        "price_momentum_1h",
        "volume_spike",
        "price_volatility",
        "rsi_14",
    ]

    df_clean = df.dropna(subset=feature_cols)

    print(f"\n✅ ML-Ready Dataset Prepared")
    print(f"   Total rows: {len(df_clean):,}")
    print(f"   Features: {len(feature_cols)}")

    if not include_unresolved:
        print(
            f"   Labels: {df_clean['label_yes'].sum()} YES, {len(df_clean) - df_clean['label_yes'].sum()} NO"
        )

    print(f"   Quality filter: ≥{min_quality}")

    if output_path:
        df_clean.to_csv(output_path, index=False)
        print(f"   Saved to: {output_path}")

    conn.close()
    return df_clean


def main():
    parser = argparse.ArgumentParser(
        description="Query and export Bitcoin market data for ML training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db", type=Path, default=DB_PATH, help=f"Database path (default: {DB_PATH})"
    )

    parser.add_argument("--stats", action="store_true", help="Show database statistics")

    parser.add_argument(
        "--export", choices=["csv", "json"], help="Export all data to CSV or JSON"
    )

    parser.add_argument(
        "--market", type=str, help="Show history for specific market ID"
    )

    parser.add_argument(
        "--ml-ready",
        action="store_true",
        help="Export ML-ready dataset (features + labels)",
    )

    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum data quality score for ML dataset (default: 0.5)",
    )

    parser.add_argument(
        "--include-unresolved",
        action="store_true",
        help="Include unresolved markets in ML dataset",
    )

    parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()

    if not args.db.exists():
        print(f"❌ Database not found: {args.db}")
        print(
            f"   Run the tracker first: python -m polymarket_agents.services.bitcoin_tracker"
        )
        return

    if args.stats:
        get_stats(args.db)

    elif args.export:
        export_data(args.db, format=args.export, output_path=args.output)

    elif args.market:
        get_market_history(args.db, args.market)

    elif args.ml_ready:
        get_ml_ready_dataset(
            args.db,
            min_quality=args.min_quality,
            include_unresolved=args.include_unresolved,
            output_path=args.output or "bitcoin_ml_dataset.csv",
        )

    else:
        # Default: show stats
        get_stats(args.db)


if __name__ == "__main__":
    main()

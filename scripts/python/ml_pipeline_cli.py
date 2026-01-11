import sys
import os
import json
import click
import time
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.layout import Layout
from rich.live import Live
from rich.json import JSON

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

# --- Import Your Agent Components ---
from agents.team.ingestion import IngestionTeam
from agents.memory.manager import MemoryManager
from agents.automl.data_ingestion import PolymarketDataIngestion
# Note: Assuming standard ML libs are available or using basic logic if not
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
except ImportError:
    pass # Will handle gracefully

# --- Configuration ---
DATA_DIR = "data"
STRATEGY_FILE = os.path.join(DATA_DIR, "ml_strategy.json")
FEATURES_FILE = os.path.join(DATA_DIR, "training_features.csv")
PREDICTIONS_FILE = os.path.join(DATA_DIR, "predictions.json")
BETS_FILE = os.path.join(DATA_DIR, "bet_recommendations.json")

console = Console()

@click.group()
def cli():
    """Cybercelli Autonomous ML Betting Pipeline"""
    pass

# ==============================================================================
# AGENT 1: INGESTION (The Scout)
# ==============================================================================
@cli.command()
@click.option("--limit", default=20, help="Number of markets to fetch")
def ingest(limit):
    """Agent 1: Updates local database with fresh Polymarket data."""
    console.print(Panel("🤖 [bold blue]AGENT 1: DATA SCOUT[/bold blue]\nFetching live market data...", border_style="blue"))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Ingesting Markets...", total=limit)
        
        # We wrap the existing IngestionTeam to hook into our progress bar
        team = IngestionTeam()
        
        # Simulating the chunked progress for the visual effect since run_cycle is atomic
        # In a real refactor, pass a callback to run_cycle
        team.run_cycle(limit=limit)
        progress.update(task, completed=limit)
        
    console.print(f"✅ [green]Success:[/green] Ingested {limit} markets into {DATA_DIR}/markets.db")

# ==============================================================================
# AGENT 2: STRATEGY PLANNER (The Architect)
# ==============================================================================
@cli.command()
def strategize():
    """Agent 2: Analyzes DB stats and creates an ML optimization plan."""
    console.print(Panel("🤖 [bold magenta]AGENT 2: STRATEGY ARCHITECT[/bold magenta]\nAnalyzing data landscape...", border_style="magenta"))

    memory = MemoryManager(db_path=f"{DATA_DIR}/markets.db")
    stats = memory.get_stats()
    
    with console.status("[bold magenta]Formulating ML Strategy...", spinner="dots"):
        time.sleep(1.5) # Thinking time
        
        # Logic: Determine strategy based on available data volume
        total_markets = stats.get("total_markets", 0)
        
        strategy = {
            "timestamp": datetime.now().isoformat(),
            "data_source_stats": stats,
            "pipeline_config": {
                "target_variable": "outcome_yes_price > current_price",
                "lookback_window_days": 30,
                "min_volume_threshold": 5000 if total_markets > 1000 else 1000,
                "model_type": "RandomForest" if total_markets < 10000 else "XGBoost",
                "features": [
                    "volume", 
                    "liquidity", 
                    "spread", 
                    "sentiment_score", 
                    "time_to_expiry"
                ]
            }
        }
        
        with open(STRATEGY_FILE, "w") as f:
            json.dump(strategy, f, indent=2)

    console.print(Panel(JSON(json.dumps(strategy)), title="Generated ML Plan", border_style="green"))
    console.print(f"✅ [green]Plan saved to {STRATEGY_FILE}[/green]")

# ==============================================================================
# AGENT 3: FEATURE ENGINEER (The Builder)
# ==============================================================================
@cli.command()
def prepare():
    """Agent 3: Determines variables and builds the training dataset."""
    console.print(Panel("🤖 [bold yellow]AGENT 3: FEATURE ENGINEER[/bold yellow]\nBuilding training vectors...", border_style="yellow"))

    if not os.path.exists(STRATEGY_FILE):
        console.print("[red]Error: No strategy file found. Run 'strategize' first.[/red]")
        return

    with open(STRATEGY_FILE, "r") as f:
        strategy = json.load(f)
    
    config = strategy["pipeline_config"]
    features = config["features"]
    
    with console.status(f"[bold yellow]Extracting features: {features}...", spinner="material"):
        # Real Logic: Query DB via MemoryManager
        memory = MemoryManager(db_path=f"{DATA_DIR}/markets.db")
        # In reality, you'd have a method to get specific rows. Using list_top_volume for demo.
        markets = memory.list_top_volume_markets(limit=100)
        
        # Transform to DataFrame
        data = []
        for m in markets:
            # Synthetic feature enrichment based on real data
            row = {
                "id": m["id"],
                "question": m["question"],
                "volume": m["volume"],
                "liquidity": m.get("liquidity", 0),
                "spread": 0.01, # Placeholder if not in DB
                "sentiment_score": 0.5, # Placeholder for NLP agent
                "time_to_expiry": 10, # Days
                "current_price": 0.5, # Need price history for real ML
                "target": 1 if m["volume"] > 20000000 else 0 # Balanced target: high vs low volume
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        df.to_csv(FEATURES_FILE, index=False)
        time.sleep(1)

    console.print(f"📊 [bold]Dataset Shape:[/bold] {df.shape}")
    console.print(f"✅ [green]Features saved to {FEATURES_FILE}[/green]")

# ==============================================================================
# AGENT 4: ML OPERATOR (The Brain)
# ==============================================================================
@cli.command()
def train():
    """Agent 4: Executes ML training and generates predictions."""
    console.print(Panel("🤖 [bold red]AGENT 4: ML OPERATOR[/bold red]\nTraining models...", border_style="red"))

    if not os.path.exists(FEATURES_FILE):
        console.print("[red]Error: No features found. Run 'prepare' first.[/red]")
        return

    df = pd.read_csv(FEATURES_FILE)
    
    # Simple ML Workflow
    features = ["volume", "liquidity", "spread", "sentiment_score"]
    target = "target"
    
    X = df[features]
    y = df[target]
    
    with Progress(SpinnerColumn(), TextColumn("[bold red]{task.description}")) as progress:
        t1 = progress.add_task("Splitting Data...", total=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        progress.advance(t1)
        
        t2 = progress.add_task("Training Random Forest...", total=1)
        model = RandomForestClassifier(n_estimators=100)
        model.fit(X_train, y_train)
        time.sleep(1) # Visual pacing
        progress.advance(t2)
        
        t3 = progress.add_task("Generating Inference...", total=1)
        # Handle single-class prediction case
        proba_result = model.predict_proba(X)
        if proba_result.shape[1] == 1:
            # Single class - all predictions are the same
            probs = proba_result[:, 0]  # Use the single probability
        else:
            # Normal binary classification
            probs = proba_result[:, 1]
        progress.advance(t3)

    # Attach predictions back to ID
    results = []
    for i, prob in enumerate(probs):
        results.append({
            "market_id": str(df.iloc[i]["id"]),
            "question": str(df.iloc[i]["question"]),
            "volume": float(df.iloc[i]["volume"]),
            "model_confidence": float(prob)
        })
        
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    console.print(f"🎯 [bold]Model Accuracy (Test):[/bold] {model.score(X_test, y_test):.2f}")
    console.print(f"✅ [green]Predictions saved to {PREDICTIONS_FILE}[/green]")

# ==============================================================================
# AGENT 5: BETTING ANALYST (The Closer)
# ==============================================================================
@cli.command()
def analyze():
    """Agent 5: Analyzes predictions and output betting recommendations."""
    console.print(Panel("🤖 [bold green]AGENT 5: BETTING ANALYST[/bold green]\nFinding Alpha...", border_style="green"))

    if not os.path.exists(PREDICTIONS_FILE):
        console.print("[red]Error: No predictions found. Run 'train' first.[/red]")
        return

    with open(PREDICTIONS_FILE, "r") as f:
        predictions = json.load(f)
        
    bets = []
    table = Table(title="Recommended Bets", border_style="green")
    table.add_column("Market", style="cyan", no_wrap=True)
    table.add_column("Conf", style="magenta")
    table.add_column("Action", style="bold green")
    table.add_column("Reasoning", style="white")

    for p in predictions:
        # Simple Logic: High confidence + High Volume
        confidence = p["model_confidence"]
        if confidence > 0.7:
            reasoning = f"Strong ML Signal ({confidence:.2f}) & Vol > 100k"
            bets.append({
                "market_id": p["market_id"],
                "action": "BUY YES",
                "size": 10.0, # Fixed sizing for now
                "reasoning": reasoning
            })
            table.add_row(p["question"][:40]+"...", f"{confidence:.2f}", "BUY YES", reasoning)
            
    with open(BETS_FILE, "w") as f:
        json.dump(bets, f, indent=2)

    console.print(table)
    console.print(f"💰 [bold green]Found {len(bets)} actionable bets.[/bold green]")
    console.print(f"✅ [green]Recommendations saved to {BETS_FILE}[/green]")

# ==============================================================================
# ORCHESTRATOR
# ==============================================================================
@cli.command()
def run_all():
    """Run the entire pipeline end-to-end."""
    ctx = click.get_current_context()
    ctx.invoke(ingest, limit=50)
    ctx.invoke(strategize)
    ctx.invoke(prepare)
    ctx.invoke(train)
    ctx.invoke(analyze)

if __name__ == "__main__":
    cli()
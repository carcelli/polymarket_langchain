#!/usr/bin/env python3
"""
Script to run the Polymarket Data Ingestion Team.
This script initializes the IngestionTeam and runs a data collection cycle.
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from agents.team.ingestion import IngestionTeam


def main():
    print("Initializing Polymarket Ingestion Team...")
    team = IngestionTeam()

    try:
        # Default to 5 markets, can be adjusted
        team.run_cycle(limit=5)
    except KeyboardInterrupt:
        print("\nIngestion stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

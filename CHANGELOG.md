# Changelog

All notable changes to this project will be documented in this file.

## [2026-01-12] - Production Refactor

### 🏗️ Architecture & Refactoring
- **Package Structure**: Migrated codebase to `src/polymarket_agents/` layout for better packaging and distribution.
    - Centralized core logic in `src/polymarket_agents/`.
    - Organized submodules: `automl`, `backends`, `connectors`, `graph`, `langchain`, `memory`, `ml_strategies`, `subagents`.
- **Configuration**: Introduced `src/polymarket_agents/config.py` for centralized, secure configuration management.
- **Dependency Management**: Added `pyproject.toml` for modern Python project definition.

### 🧹 Security & Housekeeping
- **Data Cleanup**: Removed all committed binary data artifacts (`*.db`, `*.pkl`, `*.parquet`) and large JSON dumps from `data/` and `agent_data/`.
- **Gitignore Rules**: Updated `.gitignore` to enforce exclusion of `data/`, `agent_data/`, and generated artifacts (`.langgraph_api/`, graphs).
- **Credentials**: Updated `.env.example` to reflect secure key loading practices (supporting file-based keys via `config.py`).

### 🛠️ Developer Experience
- **Script Execution**: Added executable permissions (`chmod +x`) to all shell scripts and entry points:
    - `quick_start.sh`
    - `scripts/bash/*.sh`
    - `scripts/bet_planner.sh`
    - `scripts/polymarket_agent.sh`
    - `scripts/python/run_ingestion_team.py`
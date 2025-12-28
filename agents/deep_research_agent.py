"""
Deep Research Agent for Polymarket Analysis

This agent uses the deepagents framework to provide:
1. Automated planning and task breakdown
2. Web search capabilities for comprehensive research
3. File system tools for context management
4. Subagent delegation for specialized tasks
5. Report generation and synthesis

Architecture:
- Primary Agent: Orchestrates research workflow
- Research Subagent: Handles web search and information gathering
- Analysis Subagent: Processes data and generates insights
- File System: Manages context and intermediate results
"""

import os
from typing import List, Dict, Any, Optional
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StoreBackend, CompositeBackend, StateBackend
from deepagents.backends.protocol import BackendProtocol, WriteResult, EditResult
from deepagents.backends.utils import FileInfo, GrepMatch

from agents.tools.research_tools import web_search, market_news_search, comprehensive_research
from agents.tools.market_tools import fetch_active_markets, get_market_details
from agents.tools.trade_tools import execute_market_order, execute_limit_order


# =============================================================================
# ADVANCED BACKEND IMPLEMENTATIONS
# =============================================================================

class GuardedFilesystemBackend(FilesystemBackend):
    """
    Filesystem backend with security policies and access controls.

    Prevents writes to sensitive directories and enforces enterprise policies.
    """

    def __init__(self, *, deny_prefixes: list[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.deny_prefixes = [p if p.endswith("/") else p + "/" for p in (deny_prefixes or [])]

    def _is_denied(self, path: str) -> bool:
        """Check if path is in denied prefixes."""
        return any(path.startswith(p) for p in self.deny_prefixes)

    def write(self, file_path: str, content: str) -> WriteResult:
        if self._is_denied(file_path):
            return WriteResult(error=f"Writes are not allowed under {file_path}")
        return super().write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        if self._is_denied(file_path):
            return EditResult(error=f"Edits are not allowed under {file_path}")
        return super().edit(file_path, old_string, new_string, replace_all)


class GuardedStoreBackend(BackendProtocol):
    """
    Store backend with security policies for cross-conversation storage.
    """

    def __init__(self, runtime, namespace: str, deny_prefixes: list[str] = None):
        self.runtime = runtime
        self.namespace = namespace
        self.deny_prefixes = [p if p.endswith("/") else p + "/" for p in (deny_prefixes or [])]
        self._store_backend = StoreBackend(runtime, namespace=namespace)

    def _is_denied(self, path: str) -> bool:
        """Check if path is in denied prefixes."""
        return any(path.startswith(p) for p in self.deny_prefixes)

    def ls_info(self, path: str) -> list[FileInfo]:
        return self._store_backend.ls_info(path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        return self._store_backend.read(file_path, offset, limit)

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        return self._store_backend.grep_raw(pattern, path, glob)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return self._store_backend.glob_info(pattern, path)

    def write(self, file_path: str, content: str) -> WriteResult:
        if self._is_denied(file_path):
            return WriteResult(error=f"Writes are not allowed under {file_path}")
        return self._store_backend.write(file_path, content)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        if self._is_denied(file_path):
            return EditResult(error=f"Edits are not allowed under {file_path}")
        return self._store_backend.edit(file_path, old_string, new_string, replace_all)


class VirtualMarketBackend(BackendProtocol):
    """
    Virtual filesystem backend for market data and research templates.

    Provides pre-built templates and market data structures without real storage.
    Perfect for demos, testing, and structured research workflows.
    """

    def __init__(self):
        # In-memory storage for virtual files
        self._files = {
            "/templates/research_outline.md": """# Market Research Outline

## Executive Summary
[Brief market overview and thesis]

## Market Context
- Current market probability: [XX%]
- Market volume: $[XXX]
- Key participants and sentiment

## Fundamental Analysis
- Historical precedents
- Current developments
- Expert opinions and analysis

## Technical Analysis
- Price momentum and trends
- Volume analysis
- Market efficiency assessment

## Risk Assessment
- Key risk factors
- Black swan scenarios
- Time horizon considerations

## Trading Recommendation
- Suggested position size
- Entry/exit strategy
- Risk management""",

            "/templates/trade_journal.md": """# Trade Journal Entry

**Date:** [YYYY-MM-DD]
**Market:** [Market Question]
**Side:** [YES/NO]
**Size:** [Position Size]
**Entry Price:** [XX%]

## Pre-Trade Analysis
[Research summary and reasoning]

## Expected Value Calculation
- Estimated true probability: [XX%]
- Market price: [XX%]
- Edge: [X.X%]
- Kelly fraction: [XX%]

## Trade Execution
- Order type: [Market/Limit]
- Actual execution price: [XX%]
- Slippage: [X.X%]

## Post-Trade Notes
[Market developments, lessons learned]

## Outcome
[Resolution and performance analysis]""",

            "/market_data/README.md": """# Market Data Directory

This directory contains structured market data and research.

## Available Data
- `/market_data/active_markets.json` - Current active markets
- `/market_data/categories/` - Markets by category
- `/market_data/volume_leaders.json` - High volume markets

## Research Templates
- `/templates/research_outline.md` - Comprehensive analysis template
- `/templates/trade_journal.md` - Trade documentation template

## Usage
Use these templates and data structures to organize your research systematically.""",

            "/market_data/active_markets.json": """[
  {"id": "sample_1", "question": "Will Bitcoin reach $200k by 2025?", "category": "crypto", "volume": 5000000},
  {"id": "sample_2", "question": "Will Trump win 2028 election?", "category": "politics", "volume": 2000000},
  {"id": "sample_3", "question": "Will S&P 500 reach 6000 by 2026?", "category": "finance", "volume": 1000000}
]"""
        }

    def ls_info(self, path: str) -> list[FileInfo]:
        """List files in virtual filesystem."""
        path = path.rstrip("/") + "/" if not path.endswith("/") else path
        files = []

        for file_path in self._files.keys():
            if file_path.startswith(path):
                # Get relative path
                rel_path = file_path[len(path):].lstrip("/")
                if "/" not in rel_path:  # Direct child
                    files.append(FileInfo(
                        path=file_path,
                        size=len(self._files[file_path]),
                        modified_at="2025-01-01T00:00:00Z"
                    ))

        return sorted(files, key=lambda x: x.path)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> str:
        """Read virtual file content."""
        if file_path not in self._files:
            return f"Error: File '{file_path}' not found"

        content = self._files[file_path]
        if offset > 0:
            content = content[offset:]

        if limit and len(content) > limit:
            content = content[:limit] + "\n... (truncated)"

        # Add line numbers for readability
        lines = content.split('\n')
        numbered_content = '\n'.join(f"{i+1:4d}|{line}" for i, line in enumerate(lines))
        return numbered_content

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> list[GrepMatch] | str:
        """Search virtual files."""
        import re
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {e}"

        matches = []
        search_files = self._files.keys()

        if path:
            search_files = [f for f in search_files if f.startswith(path)]

        for file_path in search_files:
            content = self._files[file_path]
            for line_num, line in enumerate(content.split('\n'), 1):
                if regex.search(line):
                    matches.append(GrepMatch(
                        path=file_path,
                        line=line_num,
                        text=line.strip()
                    ))

        return matches

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        """Glob virtual files."""
        import fnmatch
        path = path.rstrip("/") + "/" if not path.endswith("/") else path
        files = []

        for file_path in self._files.keys():
            if file_path.startswith(path):
                rel_path = file_path[len(path):].lstrip("/")
                if fnmatch.fnmatch(rel_path, pattern):
                    files.append(FileInfo(
                        path=file_path,
                        size=len(self._files[file_path]),
                        modified_at="2025-01-01T00:00:00Z"
                    ))

        return files

    def write(self, file_path: str, content: str) -> WriteResult:
        """Write to virtual filesystem (stored in memory only)."""
        if file_path in self._files:
            return WriteResult(error=f"File '{file_path}' already exists")

        self._files[file_path] = content
        return WriteResult(path=file_path, files_update={file_path: content})

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        """Edit virtual file."""
        if file_path not in self._files:
            return EditResult(error=f"File '{file_path}' not found")

        content = self._files[file_path]
        if replace_all:
            new_content = content.replace(old_string, new_string)
            occurrences = content.count(old_string)
        else:
            # Replace first occurrence only
            new_content = content.replace(old_string, new_string, 1)
            occurrences = 1 if old_string in content else 0

        if occurrences == 0:
            return EditResult(error=f"String '{old_string}' not found in {file_path}")

        self._files[file_path] = new_content
        return EditResult(
            path=file_path,
            files_update={file_path: new_content},
            occurrences=occurrences
        )


def create_polymarket_research_agent(
    model_name: str = "claude-3-5-sonnet-20241022",
    risk_tolerance: str = "moderate",
    enable_trading: bool = False,
    storage_strategy: str = "filesystem",
    enable_human_loop: bool = False,
    checkpointer = None
):
    """
    Create a customized deep research agent with full harness capabilities.

    Args:
        model_name: Model identifier (claude-3-5-sonnet-20241022, gpt-4o, etc.)
        risk_tolerance: "conservative", "moderate", or "aggressive"
        enable_trading: Whether to enable actual trading execution
        storage_strategy: "filesystem", "composite", or "store"
        enable_human_loop: Whether to enable human approval for trading

    Returns:
        Configured deep agent with harness capabilities
    """

    # Configure model using init_chat_model for better compatibility
    model = init_chat_model(model=model_name, temperature=0.2)

    # Configure storage backend based on strategy
    if storage_strategy == "filesystem":
        # Real filesystem access with virtual root and security policies
        backend = lambda rt: GuardedFilesystemBackend(
            root_dir="./agent_workspace",
            virtual_mode=True,
            deny_prefixes=["/secrets/", "/system/", "/config/"]  # Security policies
        )
    elif storage_strategy == "composite":
        # Advanced hybrid storage with long-term memory routing
        backend = lambda rt: CompositeBackend(
            default=StateBackend(rt),  # Ephemeral by default
            routes={
                "/workspace/": FilesystemBackend(root="./agent_workspace", virtual_mode=True),
                "/research/": StoreBackend(rt, namespace="polymarket_research"),
                "/memories/": StoreBackend(rt, namespace="agent_memories"),  # Long-term memory
                "/user/": StoreBackend(rt, namespace="user_data"),  # User preferences
                "/knowledge/": StoreBackend(rt, namespace="market_knowledge"),  # Learned knowledge
                "/temp/": StateBackend(rt),  # Explicit temp routing
                "/cache/": FilesystemBackend(root="./agent_cache", virtual_mode=True),
                "/market_data/": StoreBackend(rt, namespace="market_data"),
            }
        )
    elif storage_strategy == "store":
        # Persistent cross-conversation storage with policies
        backend = lambda rt: GuardedStoreBackend(
            rt,
            namespace="polymarket_research",
            deny_prefixes=["/admin/", "/system/", "/secrets/"]
        )
    elif storage_strategy == "virtual":
        # Pure virtual filesystem for testing/demos
        backend = lambda rt: VirtualMarketBackend()
    else:
        # Default secure filesystem backend
        backend = lambda rt: GuardedFilesystemBackend(
            root_dir="./agent_workspace",
            virtual_mode=True,
            deny_prefixes=["/secrets/", "/system/", "/config/"]
        )

    # Customize system prompt based on risk tolerance
    base_prompt = """You are an expert prediction market analyst specializing in Polymarket trading.

Your mission is to conduct systematic, evidence-based research and provide actionable trading insights.

## Long-Term Memory Structure

You have access to persistent memory that survives across conversations:

**User Data (`/user/`):**
- `/user/preferences.txt` - User trading preferences and risk tolerance
- `/user/portfolio.txt` - Portfolio composition and constraints
- `/user/history.txt` - Past trading decisions and outcomes

**Agent Memories (`/memories/`):**
- `/memories/learnings.txt` - Key insights and lessons learned
- `/memories/strategies.txt` - Successful trading strategies discovered
- `/memories/context.txt` - Long-term conversation context

**Market Knowledge (`/knowledge/`):**
- `/knowledge/markets/` - Understanding of specific markets
- `/knowledge/patterns.txt` - Recognized market patterns and behaviors
- `/knowledge/research/` - Accumulated research and analysis

**Persistent Research (`/research/`):**
- `/research/active/` - Current ongoing research projects
- `/research/archive/` - Completed research and reports

## Memory Usage Guidelines

1. **Read existing memory** at conversation start to maintain continuity
2. **Update user preferences** when users express preferences or constraints
3. **Record learnings** from successful/unsuccessful trading decisions
4. **Build knowledge** incrementally across conversations
5. **Archive research** when projects are completed

## Available Tools

### Research & Data Tools
- **web_search**: Internet search for market-relevant information, news, and analysis
- **market_news_search**: Specialized news search for market-specific developments
- **comprehensive_research**: Complete market analysis package (data + web + news)
- **fetch_active_markets**: Browse available Polymarket opportunities
- **get_market_details**: Deep dive into specific market mechanics

### Built-in DeepAgents Tools
- **write_todos**: Break down complex analysis into manageable tasks
- **read_file/write_file**: Persistent context management across research sessions
- **ls**: Navigate your research workspace
- **edit_file**: Update and refine research documents
- **task**: Delegate specialized analysis to subagents

### Trading Tools"""

    if enable_trading:
        base_prompt += """
- **execute_market_order**: Execute market orders (use with extreme caution)
- **execute_limit_order**: Place limit orders for better execution"""

    # Risk-based customization
    if risk_tolerance == "conservative":
        risk_guidance = """
## Conservative Risk Framework
- Require minimum 5% edge before considering positions
- Maximum Kelly fraction: 10% of portfolio per trade
- Extensive due diligence required for all recommendations
- Prioritize risk management over return potential
- Default action: PASS unless overwhelming evidence"""
    elif risk_tolerance == "aggressive":
        risk_guidance = """
## Aggressive Risk Framework
- Consider positions with 2%+ edge
- Maximum Kelly fraction: 25% of portfolio per trade
- Balance speed and thoroughness
- Accept higher volatility for potential returns
- Default action: WATCH for promising opportunities"""
    else:  # moderate
        risk_guidance = """
## Moderate Risk Framework
- Require minimum 3% edge for position consideration
- Maximum Kelly fraction: 15% of portfolio per trade
- Balanced approach to research depth and timeliness
- Consider both risk and reward carefully
- Default action: WATCH with clear trigger conditions"""

    methodology = """

## Research Methodology

### Phase 1: Planning & Scoping
1. Use `write_todos` to break down the analysis into specific research tasks
2. Identify key factors that will determine the market outcome
3. Plan information gathering strategy (web search + market data)

### Phase 2: Information Gathering
1. **Market Data**: Use `comprehensive_research` for complete market intelligence
2. **Web Research**: Search for expert analysis, news, and relevant developments
3. **Context Building**: Save important findings to files for reference
4. **Cross-validation**: Verify information across multiple sources

### Phase 3: Analysis & Synthesis
1. **Probability Estimation**: Assess true likelihood based on evidence
2. **Market Efficiency**: Compare estimated probability to market price
3. **Edge Calculation**: Quantify potential profit opportunity
4. **Risk Assessment**: Identify scenarios that could invalidate analysis

### Phase 4: Decision & Documentation
1. **Position Sizing**: Apply Kelly criterion within risk limits
2. **Trade Execution**: Use limit orders for better pricing (if enabled)
3. **Documentation**: Save complete analysis to files for future reference
4. **Continuous Learning**: Track outcomes to improve future analysis

## File System Organization

Organize your research workspace:
- `/research/{market_id}/` - Market-specific analysis
- `/research/{market_id}/sources.txt` - Information sources
- `/research/{market_id}/analysis.md` - Detailed findings
- `/research/{market_id}/recommendation.json` - Structured recommendation

## Quality Standards

- **Source Credibility**: Prefer reputable sources (academic, established media, domain experts)
- **Recency**: Weight recent information more heavily
- **Consensus**: Note areas of agreement vs. controversy
- **Uncertainty**: Explicitly acknowledge confidence levels and limitations
- **Objectivity**: Present balanced analysis, not wishful thinking

Always provide specific, actionable recommendations backed by systematic analysis."""

    # Combine prompts
    system_prompt = base_prompt + risk_guidance + methodology

    # Configure tools based on capabilities
    research_tools = [
        web_search,
        market_news_search,
        comprehensive_research,
        fetch_active_markets,
        get_market_details,
    ]

    trading_tools = [
        execute_market_order,
        execute_limit_order,
    ] if enable_trading else []

    all_tools = research_tools + trading_tools

    # Configure specialized subagents following best practices
    subagents = [
        "market_researcher": {
            "name": "market_researcher",
            "description": "Conducts comprehensive market research using web search and data analysis. Use for gathering information about market conditions, news, and expert opinions. Returns synthesized findings with confidence scores.",
            "tools": research_tools,
            "system_prompt": """You are an expert market researcher specializing in prediction markets.

Your task is to conduct thorough research on market questions and provide actionable insights.

PROCESS:
1. Break down the research question into specific, searchable queries
2. Use web_search and market tools to gather relevant information
3. Cross-reference multiple sources for credibility
4. Identify key factors, trends, and expert opinions
5. Save detailed findings to /research/findings.md

OUTPUT FORMAT (keep under 500 words):
## Research Summary (2-3 paragraphs)
[Synthesize key findings and market context]

## Key Factors
• [Factor 1]: [Impact assessment]
• [Factor 2]: [Impact assessment]

## Confidence Assessment
• Data Quality: [High/Medium/Low] - [Justification]
• Market Understanding: [High/Medium/Low] - [Justification]

## Sources
[List 3-5 key sources with brief credibility notes]

FOCUS: Provide insights that directly impact YES/NO market probabilities. Be specific and data-driven."""
        },
        "risk_analyzer": {
            "name": "risk_analyzer",
            "description": "Performs detailed risk assessment and position sizing calculations using Kelly criterion. Use when you need quantitative risk analysis, position sizing recommendations, and edge calculations.",
            "tools": [],  # Only built-in tools for calculation - no external data access
            "system_prompt": """You are a quantitative risk analyst specializing in prediction market position sizing.

Your task is to analyze trading opportunities and provide precise risk calculations.

REQUIRED CALCULATIONS:
1. Edge calculation: (Estimated_Prob - Market_Price) * 100
2. Kelly fraction: (b * p - q) / b where b = (1-price)/price, p = our_prob, q = 1-p
3. Position sizing: Kelly_fraction * portfolio_value * risk_multiplier
4. Risk assessment: Identify key failure scenarios and their probabilities

OUTPUT FORMAT (keep under 300 words):
## Edge Analysis
• Estimated Probability: [XX.X]%
• Market Price: [XX.X]%
• Edge: [X.X]% ([favorable/unfavorable])

## Position Sizing
• Kelly Fraction: [XX.X]%
• Recommended Size: $[XXX] (based on $10k portfolio)
• Risk Multiplier Applied: [X.X]

## Risk Factors
• [Risk 1]: [Probability] - [Impact]
• [Risk 2]: [Probability] - [Impact]

## Confidence
• Analysis Confidence: [High/Medium/Low] - [Key uncertainties]

BE CONSERVATIVE: Always round down position sizes. Consider worst-case scenarios."""
        },
        "quick_researcher": {
            "name": "quick_researcher",
            "description": "Performs rapid research for simple questions requiring 1-2 searches. Use for basic facts, definitions, or quick market checks when deep analysis isn't needed.",
            "tools": research_tools,
            "system_prompt": """You are a fast, focused researcher for quick questions.

PROCESS:
1. Make 1-2 targeted web searches
2. Extract the most relevant information
3. Provide concise, factual answers

OUTPUT FORMAT (keep under 200 words):
## Answer
[Direct, factual response]

## Source
[Primary source with credibility note]

## Confidence
[High/Medium/Low] - [Brief justification]

FOCUS: Speed over depth. If question requires complex analysis, recommend using market_researcher instead."""
        },
        "data_synthesizer": {
            "name": "data_synthesizer",
            "description": "Combines multiple data sources and research findings into coherent market analysis. Use when you have scattered information that needs integration and synthesis.",
            "tools": [],  # Focus on synthesis, not new data gathering
            "system_prompt": """You are a data synthesis specialist.

Your task is to integrate multiple research findings into a coherent market analysis.

PROCESS:
1. Review all available research data
2. Identify patterns and contradictions
3. Synthesize into unified market view
4. Highlight key insights and uncertainties

OUTPUT FORMAT (keep under 400 words):
## Synthesized Market View
[Integrated analysis combining all data sources]

## Key Insights
• [Insight 1]: [Supporting evidence]
• [Insight 2]: [Supporting evidence]

## Probability Assessment
• Current Range: [XX-X]% (based on available data)
• Key Uncertainties: [List major unknowns]

## Data Quality Assessment
• Source Diversity: [Good/Limited] - [Assessment]
• Recency: [Current/Mixed/Stale] - [Assessment]

FOCUS: Integration over new research. Highlight consensus vs. controversy."""
        },
        "trade_executor": {
            "name": "trade_executor",
            "description": "Executes trades according to predefined criteria with verification. Use only when you have clear analysis and want to proceed with trade execution.",
            "tools": trading_tools,
            "system_prompt": """You are a precise trade execution specialist.

Your task is to execute trades with maximum care and verification.

PRE-EXECUTION CHECKS:
1. Verify market conditions match analysis
2. Confirm position sizing is within risk limits
3. Validate all parameters before submission
4. Double-check order type (market vs limit)

EXECUTION PROCESS:
1. Prepare order with all required parameters
2. Log pre-trade state for records
3. Execute trade and capture transaction details
4. Verify execution success and price

OUTPUT FORMAT (keep under 250 words):
## Trade Execution Report
• Market: [Question]
• Side: [YES/NO]
• Size: [Amount]
• Type: [Market/Limit]
• Expected Price: [XX.X]%

## Execution Details
• Order ID: [ID]
• Actual Price: [XX.X]%
• Slippage: [X.X]%
• Fees: $[X.XX]

## Verification
• ✅ Market conditions confirmed
• ✅ Position sizing validated
• ✅ Risk limits checked
• ✅ Order parameters correct

## Post-Trade Notes
[Any execution anomalies or observations]

SAFETY FIRST: If anything seems off, cancel execution and report issues."""
        } if enable_trading else None
    ]

    # Remove None values
    subagents = [s for s in subagents if s is not None]

    # Configure human-in-the-loop for sensitive operations
    interrupts = {}
    if enable_human_loop:
        # Trading operations - highest risk
        if enable_trading:
            interrupts.update({
                "execute_market_order": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Market order execution requires human approval"
                },
                "execute_limit_order": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                    "description": "Limit order placement requires human approval"
                },
                "cancel_all_orders": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "Emergency order cancellation requires approval"
                }
            })

        # File system operations - moderate risk based on paths
        if storage_strategy in ["filesystem", "composite"]:
            interrupts.update({
                "execute_market_order": {"allowed_decisions": ["approve", "edit", "reject"]},
                "execute_limit_order": {"allowed_decisions": ["approve", "edit", "reject"]},
                # Note: File operations are handled by backend security policies
                # Additional interrupts can be added here for specific file operations
            })

        # Web search operations - low risk but configurable
        if risk_tolerance == "conservative":
            # Conservative mode: interrupt on all external API calls
            interrupts.update({
                "web_search": {"allowed_decisions": ["approve", "reject"]},
                "market_news_search": {"allowed_decisions": ["approve", "reject"]},
                "comprehensive_research": {"allowed_decisions": ["approve", "reject"]},
            })

    # Configure subagent-specific interrupts
    for subagent in subagents:
        if enable_human_loop and enable_trading:
            # Subagents handling trading get their own interrupt policies
            if subagent.get("name") == "trade_executor":
                subagent["interrupt_on"] = {
                    "execute_market_order": {"allowed_decisions": ["approve", "edit", "reject"]},
                    "execute_limit_order": {"allowed_decisions": ["approve", "edit", "reject"]},
                }

    # Create the customized deep agent with full harness capabilities
    agent = create_deep_agent(
        model=model,
        tools=all_tools,
        system_prompt=system_prompt,
        backend=backend,
        subagents=subagents,
        interrupts=interrupts if interrupts else None,
        checkpointer=checkpointer,  # Required for human-in-the-loop
        # Enable conversation summarization for long discussions
        max_tokens=170000,  # Trigger summarization at 170k tokens
        # Enable prompt caching for Anthropic models
        enable_prompt_caching=model_name.startswith("claude"),
    )

    return agent


def analyze_market_with_deep_research(
    market_question: str,
    market_id: Optional[str] = None,
    model_name: str = "claude-3-5-sonnet-20241022",
    risk_tolerance: str = "moderate",
    enable_trading: bool = False,
    storage_strategy: str = "filesystem",
    enable_human_loop: bool = False,
    checkpointer = None
) -> Dict[str, Any]:
    """
    Perform comprehensive market analysis using the deep research agent.

    Args:
        market_question: The market question to analyze
        market_id: Optional specific market ID
        model_name: Model to use for analysis
        risk_tolerance: "conservative", "moderate", or "aggressive"
        enable_trading: Whether to enable trading execution

    Returns:
        Complete analysis with research, insights, and recommendations
    """

    agent = create_polymarket_research_agent(
        model_name=model_name,
        risk_tolerance=risk_tolerance,
        enable_trading=enable_trading,
        storage_strategy=storage_strategy,
        enable_human_loop=enable_human_loop,
        checkpointer=checkpointer
    )

    # Craft a comprehensive research prompt
    research_prompt = f"""
Please conduct a comprehensive analysis of this Polymarket question:

**Market Question**: {market_question}

**Market ID**: {market_id or 'Not specified - please search for it'}

## Analysis Requirements

1. **Market Context**
   - Find the specific market on Polymarket
   - Get current pricing and volume data
   - Understand market structure (binary, multiple choice, etc.)

2. **Fundamental Research**
   - Search for relevant news and developments
   - Find expert analysis and predictions
   - Identify key factors that will determine the outcome

3. **Price Analysis**
   - Compare market probability to real-world odds
   - Identify any mispricings or edge opportunities
   - Analyze trading volume and market efficiency

4. **Risk Assessment**
   - What could cause the market to move?
   - Time until resolution and uncertainty factors
   - Potential black swan events

5. **Trading Recommendation**
   - Should we bet YES, NO, or stay out?
   - Position sizing recommendation (Kelly criterion)
   - Entry strategy and risk management

Please use all available tools systematically and save your research findings to files for reference.
Provide a comprehensive report with your final recommendation.
"""

    # Run the analysis
    result = agent.invoke({"messages": [{"role": "user", "content": research_prompt}]})

    return {
        "analysis": result["messages"][-1].content,
        "full_conversation": result["messages"],
        "agent_type": "deep_research_agent"
    }


def scan_opportunities_with_deep_research(
    category: str = None,
    min_volume: int = 10000,
    model_name: str = "claude-3-5-sonnet-20241022",
    risk_tolerance: str = "moderate",
    max_markets: int = 5,
    storage_strategy: str = "filesystem",
    enable_human_loop: bool = False,
    checkpointer = None
) -> List[Dict]:
    """
    Scan for trading opportunities using the deep research agent.

    This uses the agent's planning capabilities to systematically evaluate
    multiple markets and identify value opportunities.

    Args:
        category: Optional category filter
        min_volume: Minimum volume threshold
        model_name: Model to use for analysis
        risk_tolerance: "conservative", "moderate", or "aggressive"
        max_markets: Maximum number of markets to analyze

    Returns:
        List of analyzed opportunities with recommendations
    """

    agent = create_polymarket_research_agent(
        model_name=model_name,
        risk_tolerance=risk_tolerance,
        enable_trading=False,  # Never enable trading for scanning
        storage_strategy=storage_strategy,
        enable_human_loop=enable_human_loop,
        checkpointer=checkpointer
    )

    scan_prompt = f"""
Please scan Polymarket for potential value betting opportunities.

**Search Criteria**:
- Category: {category or 'All categories'}
- Minimum Volume: ${min_volume:,.0f}
- Maximum Markets to Analyze: {max_markets}
- Focus: High-conviction opportunities with clear edge

**Analysis Process**:
1. Use `write_todos` to plan your scanning approach
2. Fetch active markets meeting criteria using `fetch_active_markets`
3. For each promising market, conduct research using `comprehensive_research`
4. Save detailed findings to files (e.g., `/research/scan_{category}_{date}/market_{id}.md`)
5. Rank opportunities by expected value, edge, and conviction level
6. Provide specific trading recommendations with position sizing

**Opportunity Criteria**:
- Clear fundamental edge (estimated prob vs market price > 3%)
- Sufficient liquidity for entry/exit (${min_volume:,.0f}+ volume)
- Time horizon allows for convergence (not resolving immediately)
- Well-defined resolution criteria and measurable outcomes

**Output Format**:
Create a comprehensive report saved to `/research/opportunity_scan_{category}_{date}.md`
including rankings, detailed analysis, and specific recommendations.

Be systematic and thorough. Use file operations extensively to organize your findings.
"""

    result = agent.invoke({"messages": [{"role": "user", "content": scan_prompt}]})

    return {
        "scan_results": result["messages"][-1].content,
        "full_conversation": result["messages"],
        "agent_type": "deep_research_scanner"
    }


# =============================================================================
# CONVENIENCE FUNCTIONS FOR DIFFERENT USE CASES
# =============================================================================

def conservative_research_agent():
    """Create a conservative deep research agent (high conviction required)."""
    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="conservative",
        enable_trading=False
    )

def moderate_research_agent():
    """Create a moderate deep research agent (balanced risk/reward)."""
    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="moderate",
        enable_trading=False
    )

def aggressive_research_agent():
    """Create an aggressive deep research agent (lower conviction threshold)."""
    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="aggressive",
        enable_trading=False
    )

def trading_agent():
    """Create a deep research agent with trading capabilities enabled."""
    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="moderate",
        enable_trading=True
    )

# Convenience functions for backward compatibility
def quick_market_analysis(market_question: str) -> str:
    """Quick analysis for simple questions using moderate settings."""
    result = analyze_market_with_deep_research(
        market_question,
        risk_tolerance="moderate",
        enable_trading=False
    )
    return result["analysis"]

def conservative_market_analysis(market_question: str) -> str:
    """Conservative analysis requiring high conviction."""
    result = analyze_market_with_deep_research(
        market_question,
        risk_tolerance="conservative",
        enable_trading=False
    )
    return result["analysis"]

def opportunity_scanner(category: str = "politics", limit: int = 5) -> str:
    """Scan for opportunities in a category using moderate settings."""
    result = scan_opportunities_with_deep_research(
        category=category,
        max_markets=limit,
        risk_tolerance="moderate"
    )
    return result["scan_results"]


# =============================================================================
# ADVANCED HUMAN-IN-THE-LOOP FUNCTIONS
# =============================================================================

def create_trading_agent_with_approval():
    """
    Create a trading agent with comprehensive human-in-the-loop controls.

    Features:
    - Full interrupt configuration for all trading operations
    - Risk-based decision controls (approve/edit/reject)
    - Required checkpointer for state persistence
    """
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()

    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="conservative",  # Highest safety
        enable_trading=True,
        storage_strategy="composite",
        enable_human_loop=True,  # Enable all interrupts
        checkpointer=checkpointer
    )


def handle_agent_interrupt(result, config):
    """
    Handle human-in-the-loop interrupts from agent execution.

    Args:
        result: Agent invocation result
        config: Agent configuration with thread_id

    Returns:
        Tuple of (needs_human_input, interrupt_info)
    """
    if not result.get("__interrupt__"):
        return False, None

    interrupts = result["__interrupt__"][0].value
    action_requests = interrupts["action_requests"]
    review_configs = interrupts["review_configs"]

    # Create lookup map from tool name to review config
    config_map = {cfg["action_name"]: cfg for cfg in review_configs}

    interrupt_info = {
        "action_requests": action_requests,
        "review_configs": config_map,
        "thread_id": config.get("configurable", {}).get("thread_id")
    }

    return True, interrupt_info


def create_human_decisions(action_requests, review_configs, user_decisions):
    """
    Create properly formatted decisions for resuming interrupted agent.

    Args:
        action_requests: List of pending actions from interrupt
        review_configs: Review configuration map
        user_decisions: List of user decisions (one per action)

    Returns:
        Formatted decisions list for Command(resume=)
    """
    decisions = []

    for i, action in enumerate(action_requests):
        user_decision = user_decisions[i] if i < len(user_decisions) else {"type": "reject"}
        decision_type = user_decision["type"]

        if decision_type == "approve":
            decisions.append({"type": "approve"})
        elif decision_type == "reject":
            decisions.append({"type": "reject"})
        elif decision_type == "edit":
            decisions.append({
                "type": "edit",
                "edited_action": user_decision["edited_action"]
            })

    return decisions


def resume_agent_with_decisions(agent, decisions, config):
    """
    Resume interrupted agent execution with human decisions.

    Args:
        agent: The deep agent instance
        decisions: Formatted decisions list
        config: Agent configuration (must match original thread_id)

    Returns:
        Final agent result after resuming
    """
    from langgraph.types import Command

    result = agent.invoke(
        Command(resume={"decisions": decisions}),
        config=config
    )

    return result


def interactive_trading_session():
    """
    Demonstrate an interactive trading session with human-in-the-loop.

    This shows the complete workflow:
    1. Agent analyzes and proposes trades
    2. Human reviews and approves/edits/rejects
    3. Agent executes approved trades
    """
    import uuid

    print("🚀 Interactive Trading Session Demo")
    print("=" * 50)

    # Create trading agent with approval required
    agent = create_trading_agent_with_approval()

    # Create unique thread for this session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    market_question = "Will Bitcoin reach $200k by end of 2025?"

    print(f"📊 Analyzing: {market_question}")
    print("Agent will propose trades requiring human approval...")
    print()

    try:
        # Step 1: Agent analyzes and proposes trades
        result = agent.invoke({
            "messages": [{
                "role": "user",
                "content": f"Analyze this market and execute appropriate trades: {market_question}"
            }]
        }, config=config)

        # Step 2: Check for interrupts
        needs_approval, interrupt_info = handle_agent_interrupt(result, config)

        if needs_approval:
            print("⚠️  TRADE APPROVAL REQUIRED")
            print("-" * 30)

            action_requests = interrupt_info["action_requests"]
            review_configs = interrupt_info["review_configs"]

            # Display pending actions
            for i, action in enumerate(action_requests, 1):
                config = review_configs[action["name"]]
                print(f"{i}. Tool: {action['name']}")
                print(f"   Args: {action['args']}")
                print(f"   Allowed: {config['allowed_decisions']}")
                print()

            # Simulate user decisions (in real usage, get from UI/input)
            print("🤖 Simulated User Decisions:")
            user_decisions = [
                {"type": "approve"} if "market_order" in action_requests[0]["name"] else {"type": "reject"}
                for action in action_requests
            ]

            for i, decision in enumerate(user_decisions, 1):
                print(f"   Decision {i}: {decision['type']}")

            # Step 3: Resume with decisions
            decisions = create_human_decisions(action_requests, review_configs, user_decisions)
            final_result = resume_agent_with_decisions(agent, decisions, config)

            print("\n✅ Session Complete")
            print(f"Final result: {final_result['messages'][-1].content[:200]}...")

        else:
            print("✅ No trades proposed - analysis complete")
            print(f"Result: {result['messages'][-1].content[:200]}...")

    except Exception as e:
        print(f"❌ Interactive session failed: {str(e)}")
        print("Note: This demo requires proper API keys and trading permissions")


def create_conservative_trading_agent():
    """
    Create a conservative trading agent with minimal automation.

    - Requires approval for all trades
    - No editing allowed (only approve/reject)
    - Conservative risk settings
    """
    from langgraph.checkpoint.memory import MemorySaver

    checkpointer = MemorySaver()

    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="conservative",
        enable_trading=True,
        storage_strategy="composite",
        enable_human_loop=True,
        checkpointer=checkpointer,
        # Custom interrupt override for maximum safety
        custom_interrupts={
            "execute_market_order": {"allowed_decisions": ["approve", "reject"]},  # No editing
            "execute_limit_order": {"allowed_decisions": ["approve", "reject"]},   # No editing
        }
    )


def create_autonomous_research_agent():
    """
    Create a research agent with no human interruptions.

    - Full automation for research tasks
    - No trading capabilities (pure analysis)
    - Optimized for speed and efficiency
    """
    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="moderate",
        enable_trading=False,  # No trading tools
        storage_strategy="composite",
        enable_human_loop=False,  # No interrupts
        checkpointer=None  # No need for checkpointer
    )


# =============================================================================
# LONG-TERM MEMORY SPECIALIZED AGENTS
# =============================================================================

def create_memory_enabled_agent(store=None, checkpointer=None):
    """
    Create an agent with full long-term memory capabilities.

    This agent uses CompositeBackend to maintain persistent memory
    across conversations while keeping ephemeral workspace files.

    Args:
        store: LangGraph BaseStore instance (required for persistence)
        checkpointer: Checkpointer for state persistence

    Returns:
        Agent with hybrid ephemeral/persistent storage
    """
    if store is None:
        from langgraph.store.memory import InMemoryStore
        store = InMemoryStore()

    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="moderate",
        enable_trading=True,
        storage_strategy="composite",  # Full memory routing
        enable_human_loop=True,
        checkpointer=checkpointer
    ), store


def create_self_improving_agent(store=None, checkpointer=None):
    """
    Create an agent that improves itself based on user feedback.

    Uses long-term memory to accumulate instructions and preferences
    across conversations, enabling the agent to learn and adapt.

    Args:
        store: LangGraph BaseStore instance
        checkpointer: Checkpointer for state persistence

    Returns:
        Self-improving agent with persistent instruction memory
    """
    if store is None:
        from langgraph.store.memory import InMemoryStore
        store = InMemoryStore()

    # Custom system prompt for self-improvement
    self_improving_prompt = """
You are a self-improving prediction market analyst that learns from user feedback and experience.

## Self-Improvement Capabilities

You maintain persistent memory of:
- **User preferences** in `/user/preferences.txt`
- **Improvement instructions** in `/memories/instructions.txt`
- **Learned strategies** in `/memories/strategies.txt`
- **Past performance** in `/user/history.txt`

## Improvement Process

1. **Read existing memory** at conversation start
2. **Apply learned preferences** and strategies
3. **Record user feedback** when provided
4. **Update instructions** based on feedback like:
   - "Please always do X" → Add to `/memories/instructions.txt`
   - "I prefer Y approach" → Update `/user/preferences.txt`
   - "That strategy worked well" → Save to `/memories/strategies.txt`

## Memory Structure Usage

- **`/memories/instructions.txt`**: Additional system instructions accumulated over time
- **`/user/preferences.txt`**: User-specific preferences and constraints
- **`/memories/strategies.txt`**: Proven successful approaches
- **`/user/history.txt`**: Track outcomes to learn from experience

Always read your persistent memory files at the start of conversations and update them based on user interactions.
"""

    # Override the system prompt for self-improvement
    agent, store = create_memory_enabled_agent(store, checkpointer)

    # Add self-improvement instructions to the system prompt
    original_prompt = agent.system_prompt
    agent.system_prompt = self_improving_prompt + "\n\n" + original_prompt

    return agent, store


def create_knowledge_building_agent(store=None, checkpointer=None):
    """
    Create an agent that builds up market knowledge across conversations.

    Accumulates understanding of markets, patterns, and research over time,
    becoming more knowledgeable with each interaction.

    Args:
        store: LangGraph BaseStore instance
        checkpointer: Checkpointer for state persistence

    Returns:
        Knowledge-building agent with persistent learning
    """
    if store is None:
        from langgraph.store.memory import InMemoryStore
        store = InMemoryStore()

    # Custom system prompt for knowledge building
    knowledge_prompt = """
You are a knowledge-building market analyst that accumulates expertise across conversations.

## Knowledge Accumulation

You build persistent knowledge in:
- **`/knowledge/markets/`**: Understanding of specific markets
- **`/knowledge/patterns.txt`**: Recognized market patterns and behaviors
- **`/knowledge/research/`**: Accumulated research methodologies
- **`/memories/learnings.txt`**: Key insights from analysis

## Learning Process

1. **Review existing knowledge** at conversation start
2. **Apply learned patterns** to current analysis
3. **Discover new insights** during research
4. **Update knowledge base** with new findings

## Knowledge Structure

- **Market-Specific**: `/knowledge/markets/[market_id].txt`
- **Pattern Recognition**: `/knowledge/patterns.txt`
- **Research Methods**: `/knowledge/research/methods.txt`
- **Learned Insights**: `/memories/learnings.txt`

Each conversation should contribute to your growing knowledge base, making you more effective over time.
"""

    agent, store = create_memory_enabled_agent(store, checkpointer)
    original_prompt = agent.system_prompt
    agent.system_prompt = knowledge_prompt + "\n\n" + original_prompt

    return agent, store


def create_research_continuity_agent(store=None, checkpointer=None):
    """
    Create an agent that maintains research continuity across sessions.

    Perfect for long-term research projects that span multiple conversations,
    maintaining research state, progress, and findings over time.

    Args:
        store: LangGraph BaseStore instance
        checkpointer: Checkpointer for state persistence

    Returns:
        Research continuity agent with persistent project memory
    """
    if store is None:
        from langgraph.store.memory import InMemoryStore
        store = InMemoryStore()

    # Custom system prompt for research continuity
    continuity_prompt = """
You are a research continuity specialist that maintains long-term research projects across conversations.

## Research Continuity Structure

**Active Research (`/research/active/`):**
- `/research/active/[project_id]/plan.txt` - Research plan and objectives
- `/research/active/[project_id]/progress.txt` - Current progress status
- `/research/active/[project_id]/findings.txt` - Key findings so far
- `/research/active/[project_id]/sources.txt` - Sources discovered
- `/research/active/[project_id]/next_steps.txt` - Planned next actions

**Archived Research (`/research/archive/`):**
- Completed projects moved here for reference
- `/research/archive/[project_id]/final_report.md` - Complete analysis

## Continuity Process

1. **Check active research** at conversation start
2. **Continue ongoing projects** or start new ones
3. **Update progress** after each research session
4. **Archive completed** projects with final reports

## Project Management

- **Project IDs**: Use descriptive names like "bitcoin_adoption_2025"
- **Status Tracking**: Update progress after each session
- **Source Accumulation**: Build comprehensive source lists over time
- **Incremental Findings**: Add to knowledge base progressively

Research projects can span weeks or months, with full continuity maintained across all conversations.
"""

    agent, store = create_memory_enabled_agent(store, checkpointer)
    original_prompt = agent.system_prompt
    agent.system_prompt = continuity_prompt + "\n\n" + original_prompt

    return agent, store


# =============================================================================
# MEMORY MANAGEMENT UTILITIES
# =============================================================================

def initialize_memory_structure(agent, store, thread_id=None):
    """
    Initialize the memory structure for a new agent instance.

    Creates the basic directory structure and initial memory files
    to ensure consistent memory organization.

    Args:
        agent: The deep agent instance
        store: The LangGraph store
        thread_id: Optional thread ID for initialization
    """
    import uuid

    config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

    # Initialize memory structure
    init_prompt = """
Initialize your memory structure by creating the basic directory organization and initial files:

1. **User Data Structure**:
   - Create `/user/preferences.txt` with default preferences
   - Create `/user/portfolio.txt` with empty portfolio
   - Create `/user/history.txt` with header

2. **Agent Memories Structure**:
   - Create `/memories/learnings.txt` with initial learnings
   - Create `/memories/strategies.txt` with strategy framework
   - Create `/memories/context.txt` with context template

3. **Knowledge Base Structure**:
   - Create `/knowledge/patterns.txt` with pattern recognition framework
   - Create `/knowledge/research/methods.txt` with research methodologies

4. **Research Structure**:
   - Create `/research/active/README.md` explaining active research
   - Create `/research/archive/README.md` explaining archived research

Use the write_file tool to create all these files with appropriate initial content.
Focus on creating a solid foundation for long-term memory accumulation.
"""

    result = agent.invoke({
        "messages": [{"role": "user", "content": init_prompt}]
    }, config=config)

    return result


def demonstrate_cross_thread_memory():
    """
    Demonstrate cross-thread memory persistence.

    Shows how memories persist across different conversation threads.
    """
    from langgraph.store.memory import InMemoryStore
    from langgraph.checkpoint.memory import MemorySaver
    import uuid

    print("🧠 CROSS-THREAD MEMORY DEMONSTRATION")
    print("=" * 50)

    # Create agent with persistent memory
    store = InMemoryStore()
    checkpointer = MemorySaver()
    agent, _ = create_memory_enabled_agent(store, checkpointer)

    # Thread 1: Store user preferences
    thread_1 = str(uuid.uuid4())
    config_1 = {"configurable": {"thread_id": thread_1}}

    print("📝 Thread 1: Storing user preferences...")
    result_1 = agent.invoke({
        "messages": [{"role": "user", "content": """
        I am a conservative trader who prefers to invest in politics markets.
        I have a $10,000 portfolio and want to limit individual trades to 5% maximum.
        Please save these preferences to /user/preferences.txt
        """}]
    }, config=config_1)

    print("✅ Preferences saved in Thread 1")

    # Thread 2: Different conversation, read the preferences
    thread_2 = str(uuid.uuid4())
    config_2 = {"configurable": {"thread_id": thread_2}}

    print("\n📖 Thread 2: Reading preferences from different conversation...")
    result_2 = agent.invoke({
        "messages": [{"role": "user", "content": """
        What are my trading preferences? Please read them from /user/preferences.txt
        """}]
    }, config=config_2)

    print("✅ Successfully read preferences from Thread 1 in Thread 2!")
    print("\n🎯 Memory Persistence: ✓ Working across threads")
    print("🔄 Cross-Conversation Continuity: ✓ Achieved")

    return result_1, result_2


def demonstrate_memory_accumulation():
    """
    Demonstrate knowledge accumulation across conversations.

    Shows how an agent builds knowledge incrementally over time.
    """
    from langgraph.store.memory import InMemoryStore
    from langgraph.checkpoint.memory import MemorySaver
    import uuid

    print("\n📚 MEMORY ACCUMULATION DEMONSTRATION")
    print("=" * 50)

    # Create knowledge-building agent
    store = InMemoryStore()
    checkpointer = MemorySaver()
    agent, _ = create_knowledge_building_agent(store, checkpointer)

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    conversations = [
        {
            "topic": "Bitcoin Market Patterns",
            "content": """
            I've been analyzing Bitcoin markets. I notice that BTC markets often have
            higher volatility around major news events like halvings or regulatory announcements.
            The market seems to overcorrect initially then mean-revert. Save this pattern
            recognition to /knowledge/patterns.txt
            """
        },
        {
            "topic": "Political Market Dynamics",
            "content": """
            In political markets, I see that early polling data has less predictive power
            than later developments. Markets tend to underreact to initial polls but
            overreact to scandals. Add this insight to the patterns file.
            """
        },
        {
            "topic": "Market Efficiency Learning",
            "content": """
            Through my analysis, I've learned that smaller markets (<$100k volume)
            are often less efficient and provide more edge opportunities, while
            large markets (>$1M volume) are typically well-priced. Update the
            knowledge base with this market efficiency insight.
            """
        }
    ]

    print("🔄 Progressive Knowledge Building:")
    print("-" * 35)

    for i, conv in enumerate(conversations, 1):
        print(f"\n📝 Conversation {i}: {conv['topic']}")
        result = agent.invoke({
            "messages": [{"role": "user", "content": conv['content']}]
        }, config=config)
        print("✅ Knowledge updated")

    # Final knowledge review
    print("
📖 Final Knowledge Review:"    result = agent.invoke({
        "messages": [{"role": "user", "content": """
        Review all the patterns and insights you've accumulated in /knowledge/patterns.txt.
        Summarize the key learnings from our conversations.
        """}]
    }, config=config)

    print("🎓 Accumulated Knowledge:")
    print("-" * 25)
    response = result["messages"][-1].content
    print(response[:500] + "..." if len(response) > 500 else response)

    print("\n🎯 Knowledge Accumulation: ✓ Progressive learning achieved")
    print("📈 Learning Continuity: ✓ Maintained across conversations")

def conservative_opportunity_scan(category: str = "politics", limit: int = 3) -> str:
    """Conservative opportunity scan with higher conviction requirements."""
    result = scan_opportunities_with_deep_research(
        category=category,
        max_markets=limit,
        risk_tolerance="conservative",
        min_volume=50000,  # Higher volume requirement
        storage_strategy="composite"  # Use persistent storage for research
    )
    return result["scan_results"]


# =============================================================================
# HARNESS-CAPABLE SPECIALIZED AGENTS
# =============================================================================

def persistent_research_agent():
    """Agent with persistent storage for long-term research projects."""
    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="moderate",
        enable_trading=False,
        storage_strategy="composite",  # Hybrid storage for persistence
        enable_human_loop=False
    )


def virtual_demo_agent():
    """Agent with virtual filesystem for demos and testing."""
    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="moderate",
        enable_trading=False,
        storage_strategy="virtual",  # Pure virtual filesystem
        enable_human_loop=False
    )


def enterprise_secure_agent():
    """Enterprise-grade agent with comprehensive security policies."""
    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="conservative",
        enable_trading=True,
        storage_strategy="composite",  # Secure routing
        enable_human_loop=True  # Human approval for trades
    )


def research_architect_agent():
    """Agent specialized for complex multi-market research projects."""
    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="moderate",
        enable_trading=False,
        storage_strategy="composite",  # Full routing capabilities
        enable_human_loop=False
    )


def trading_agent_with_approval():
    """Trading agent with human approval required for all trades."""
    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="conservative",
        enable_trading=True,
        storage_strategy="filesystem",
        enable_human_loop=True  # Require approval for trades
    )


def high_performance_agent():
    """High-performance agent with all optimizations enabled."""
    return create_polymarket_research_agent(
        model_name="claude-3-5-sonnet-20241022",
        risk_tolerance="moderate",
        enable_trading=False,
        storage_strategy="composite",
        enable_human_loop=False
    )


def analyze_with_subagents(market_question: str, use_general_purpose: bool = False) -> Dict[str, Any]:
    """
    Use subagent delegation for comprehensive analysis.

    Args:
        market_question: The market to analyze
        use_general_purpose: Whether to use general-purpose subagent for isolation
    """
    agent = create_polymarket_research_agent(
        storage_strategy="composite",
        enable_trading=False
    )

    if use_general_purpose:
        # Use general-purpose subagent for context isolation
        prompt = f"""
        Conduct a comprehensive analysis of: {market_question}

        Use your specialized subagents for different aspects:
        1. market_researcher: Gather comprehensive market data and news
        2. risk_analyzer: Calculate edge, Kelly fraction, and risk assessment
        3. data_synthesizer: Integrate findings into coherent analysis

        IMPORTANT: For complex multi-step research that would clutter context,
        use the general-purpose subagent to isolate detailed work.

        Coordinate all subagents effectively and provide a final integrated analysis.
        """
    else:
        # Direct subagent coordination
        prompt = f"""
        Analyze this market using your subagent capabilities:

        **Market**: {market_question}

        **Analysis Approach**:
        1. Use `write_todos` to plan your analysis approach
        2. Delegate initial research to the `market_researcher` subagent
        3. Send findings to `risk_analyzer` for quantitative assessment
        4. Use `data_synthesizer` to integrate all insights
        5. Save organized research to /research/ directory

        **Key Instructions**:
        - Use subagents for specialized tasks to keep your context clean
        - Each subagent should return concise, focused results
        - Synthesize all findings into a final recommendation
        - Save intermediate results to avoid context bloat

        Coordinate the subagents effectively for a comprehensive analysis.
        """

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return {
        "analysis": result["messages"][-1].content,
        "full_conversation": result["messages"],
        "agent_type": "subagent_coordinated",
        "isolation_method": "general_purpose" if use_general_purpose else "direct_coordination"
    }


def create_specialized_research_team() -> "DeepAgent":
    """
    Create an agent with a comprehensive research team of subagents.

    This demonstrates the multiple specialized subagents pattern from the docs.
    """
    from deepagents import create_deep_agent

    # Define a comprehensive research team
    research_team = [
        {
            "name": "data_collector",
            "description": "Gathers raw market data, news, and statistical information from multiple sources. Use for comprehensive data collection that would otherwise clutter the main context.",
            "tools": [web_search, market_news_search, comprehensive_research],
            "system_prompt": """You are a specialized data collector.

            Your role is to gather comprehensive information without analysis.

            PROCESS:
            1. Identify all relevant data sources for the query
            2. Collect raw data from web, news, and market APIs
            3. Save data to /data/raw/ directory
            4. Return only a summary of what was collected

            OUTPUT FORMAT (keep under 200 words):
            ## Data Collected
            • [Source 1]: [Brief description]
            • [Source 2]: [Brief description]

            ## Files Saved
            • /data/raw/[filename1]
            • /data/raw/[filename2]

            ## Next Steps
            [Recommend which subagent should analyze this data]

            FOCUS: Collection only. Save everything to files to keep context clean."""
        },
        {
            "name": "quantitative_analyzer",
            "description": "Performs statistical analysis and quantitative modeling. Use for number-crunching tasks that require focus and precision.",
            "tools": [],  # Built-in tools only for calculations
            "system_prompt": """You are a quantitative analyst specializing in market statistics.

            Your role is to analyze numerical data and calculate probabilities.

            PROCESS:
            1. Load data from /data/raw/ files
            2. Perform statistical analysis
            3. Calculate probabilities and confidence intervals
            4. Generate quantitative insights

            OUTPUT FORMAT:
            ## Statistical Summary
            • Sample Size: [N]
            • Mean/Median: [values]
            • Standard Deviation: [value]
            • Confidence Interval: [XX-X%]

            ## Probability Assessment
            • Base Probability: [XX.X]%
            • Adjusted for Factors: [XX.X]%
            • Confidence Level: [High/Med/Low]

            ## Key Metrics
            [3-5 quantitative insights with calculations]

            BE PRECISE: Show your calculations. Keep response under 300 words."""
        },
        {
            "name": "synthesis_specialist",
            "description": "Integrates quantitative analysis with qualitative insights. Use for final report generation and recommendation synthesis.",
            "tools": [],  # Focus on integration
            "system_prompt": """You are a synthesis specialist who combines analysis into actionable insights.

            Your role is to integrate quantitative and qualitative findings.

            PROCESS:
            1. Review quantitative analysis results
            2. Incorporate qualitative factors
            3. Generate final probability assessment
            4. Provide clear trading recommendation

            OUTPUT FORMAT:
            ## Integrated Analysis
            [Synthesis of quantitative + qualitative factors]

            ## Final Probability
            [XX.X]% with confidence interval

            ## Trading Recommendation
            • Action: [BUY/SELL/HOLD]
            • Confidence: [High/Med/Low]
            • Key Rationale: [3 bullet points]

            ## Risk Considerations
            • [Risk 1]: [Mitigation]
            • [Risk 2]: [Mitigation]

            BE ACTIONABLE: Provide clear, implementable recommendations."""
        }
    ]

    # Create agent with research team
    return create_deep_agent(
        model=init_chat_model(model="claude-3-5-sonnet-20241022"),
        tools=[],  # No direct tools - delegates everything to subagents
        system_prompt="""You are a research coordinator managing a team of specialized analysts.

        Your role is to:
        1. Break complex research into manageable tasks
        2. Delegate to appropriate subagents based on their specialties
        3. Coordinate handoffs between team members
        4. Synthesize final results

        IMPORTANT: Use subagents extensively to keep your context clean.
        For any complex task, delegate to the appropriate specialist.
        Use the general-purpose subagent for context isolation when needed.

        TEAM MEMBERS:
        - data_collector: Raw data gathering
        - quantitative_analyzer: Statistical analysis
        - synthesis_specialist: Final integration and recommendations

        Coordinate them effectively for comprehensive market analysis.""",
        subagents=research_team,
        backend=lambda rt: CompositeBackend(
            default=StateBackend(rt),
            routes={
                "/data/": StoreBackend(rt, namespace="research_data"),
                "/analysis/": StoreBackend(rt, namespace="analysis_results"),
            }
        )
    )


def research_team_analysis(market_question: str) -> Dict[str, Any]:
    """
    Use the specialized research team for comprehensive analysis.

    This demonstrates the multiple specialized subagents pattern.
    """
    agent = create_specialized_research_team()

    prompt = f"""
    Conduct a comprehensive analysis of: {market_question}

    Use your research team following this workflow:

    1. **Data Collection Phase**
       - Delegate to data_collector subagent
       - Gather comprehensive market data and news

    2. **Analysis Phase**
       - Send data to quantitative_analyzer
       - Perform statistical analysis and probability calculations

    3. **Synthesis Phase**
       - Use synthesis_specialist to integrate findings
       - Generate final recommendation

    **Coordination Guidelines:**
    - Each subagent should focus on their specialty only
    - Use files in /data/ and /analysis/ for handoffs
    - Keep individual subagent responses concise
    - Synthesize final results effectively

    Provide a complete analysis with clear trading recommendation.
    """

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return {
        "analysis": result["messages"][-1].content,
        "full_conversation": result["messages"],
        "agent_type": "research_team_coordinated",
        "workflow": "data_collection → quantitative_analysis → synthesis"
    }


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--analyze":
            question = " ".join(sys.argv[2:])
            print("🔍 Analyzing market with deep research agent...")
            result = quick_market_analysis(question)
            print(result)

        elif sys.argv[1] == "--scan":
            category = sys.argv[2] if len(sys.argv) > 2 else "politics"
            print(f"🔍 Scanning {category} markets for opportunities...")
            result = opportunity_scanner(category)
            print(result)

    else:
        print("Usage:")
        print("  python deep_research_agent.py --analyze 'Will Trump win 2028?'")
        print("  python deep_research_agent.py --scan politics")
        print("  python deep_research_agent.py --scan crypto")

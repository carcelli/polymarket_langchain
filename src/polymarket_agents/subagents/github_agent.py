"""
GitHub Agent Subagent

Specialized subagent for GitHub repository management and automation.
Handles issues, PRs, commits, and repository operations.
"""

import os
from typing import List, Dict, Any
from datetime import datetime

# GitHub toolkit imports (will be available when installed)
try:
    from langchain_community.agent_toolkits.github import GitHubToolkit
    from langchain_community.utilities.github import GitHubAPIWrapper

    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False


def check_github_repo_status(repo_name: str = None) -> Dict[str, Any]:
    """
    Check the status of a GitHub repository.

    Args:
        repo_name: Repository name (format: owner/repo). If None, uses env var.
    """
    if not GITHUB_AVAILABLE:
        return {
            "error": "GitHub toolkit not installed. Run: pip install pygithub langchain-community"
        }

    try:
        repo = repo_name or os.getenv("GITHUB_REPOSITORY")
        if not repo:
            return {"error": "No repository specified. Set GITHUB_REPOSITORY env var."}

        # Get GitHub API wrapper
        github = GitHubAPIWrapper()
        api = github.github

        # Get repository info
        repo_obj = api.get_repo(repo)

        status = {
            "repository": repo,
            "name": repo_obj.name,
            "owner": repo_obj.owner.login,
            "description": repo_obj.description,
            "language": repo_obj.language,
            "stars": repo_obj.stargazers_count,
            "forks": repo_obj.forks_count,
            "open_issues": repo_obj.open_issues_count,
            "private": repo_obj.private,
            "last_updated": (
                repo_obj.updated_at.isoformat() if repo_obj.updated_at else None
            ),
            "default_branch": repo_obj.default_branch,
        }

        return status

    except Exception as e:
        return {"error": f"Failed to check repo status: {str(e)}"}


def search_github_issues(
    query: str,
    repo_name: str = None,
    state: str = "open",
    labels: List[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """
    Search for GitHub issues in a repository.

    Args:
        query: Search query
        repo_name: Repository name (format: owner/repo)
        state: Issue state ('open', 'closed', 'all')
        labels: List of label names to filter by
        limit: Maximum number of results
    """
    if not GITHUB_AVAILABLE:
        return {"error": "GitHub toolkit not installed"}

    try:
        repo = repo_name or os.getenv("GITHUB_REPOSITORY")
        if not repo:
            return {"error": "No repository specified"}

        github = GitHubAPIWrapper()
        api = github.github

        # Build search query
        search_query = f"repo:{repo} {query}"
        if state != "all":
            search_query += f" state:{state}"
        if labels:
            search_query += f" label:{','.join(labels)}"

        # Search issues
        issues = api.search_issues_and_pull_requests(
            query=search_query, sort="updated", order="desc"
        )

        results = []
        for issue in issues[:limit]:
            issue_data = {
                "number": issue.number,
                "title": issue.title,
                "state": issue.state,
                "created_at": issue.created_at.isoformat(),
                "updated_at": issue.updated_at.isoformat(),
                "author": issue.user.login if issue.user else "unknown",
                "labels": [label.name for label in issue.labels],
                "url": issue.html_url,
                "is_pull_request": issue.pull_request is not None,
            }

            # Add PR-specific data if it's a PR
            if issue.pull_request:
                try:
                    pr = api.get_pull(issue.number)
                    issue_data.update(
                        {
                            "pr_state": pr.state,
                            "mergeable": pr.mergeable,
                            "merged": pr.merged,
                            "additions": pr.additions,
                            "deletions": pr.deletions,
                            "changed_files": pr.changed_files,
                        }
                    )
                except:
                    pass

            results.append(issue_data)

        return {
            "query": query,
            "repository": repo,
            "total_results": issues.totalCount,
            "returned_results": len(results),
            "issues": results,
        }

    except Exception as e:
        return {"error": f"Failed to search issues: {str(e)}"}


def analyze_github_activity(
    repo_name: str = None, days_back: int = 30
) -> Dict[str, Any]:
    """
    Analyze GitHub repository activity over a time period.

    Args:
        repo_name: Repository name
        days_back: Number of days to analyze
    """
    if not GITHUB_AVAILABLE:
        return {"error": "GitHub toolkit not installed"}

    try:
        repo = repo_name or os.getenv("GITHUB_REPOSITORY")
        if not repo:
            return {"error": "No repository specified"}

        github = GitHubAPIWrapper()
        api = github.github

        repo_obj = api.get_repo(repo)
        since_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        since_date = since_date.replace(day=since_date.day - days_back)

        # Get recent commits
        commits = list(repo_obj.get_commits(since=since_date))
        commit_count = len(commits)

        # Get recent issues
        issues = list(repo_obj.get_issues(state="all", since=since_date))
        issue_count = len(issues)

        # Get recent PRs
        pulls = list(repo_obj.get_pulls(state="all", sort="updated", direction="desc"))
        recent_prs = [pr for pr in pulls if pr.updated_at >= since_date]
        pr_count = len(recent_prs)

        # Calculate activity metrics
        activity_score = (commit_count * 2) + (pr_count * 3) + (issue_count * 1)
        activity_level = (
            "High"
            if activity_score > 50
            else "Medium" if activity_score > 20 else "Low"
        )

        # Get contributor stats
        contributors = {}
        for commit in commits[:100]:  # Limit for performance
            author = commit.author.login if commit.author else "unknown"
            contributors[author] = contributors.get(author, 0) + 1

        analysis = {
            "repository": repo,
            "analysis_period_days": days_back,
            "activity_metrics": {
                "commits": commit_count,
                "issues_created": len(
                    [i for i in issues if i.created_at >= since_date]
                ),
                "pull_requests": pr_count,
                "total_activity_score": activity_score,
                "activity_level": activity_level,
            },
            "contributors": {
                "unique_contributors": len(contributors),
                "top_contributors": sorted(
                    contributors.items(), key=lambda x: x[1], reverse=True
                )[:5],
            },
            "insights": [
                f"Repository shows {activity_level.lower()} activity with {commit_count} commits in the last {days_back} days",
                f"{len(contributors)} unique contributors in the analysis period",
                f"{pr_count} pull requests updated recently",
            ],
        }

        return analysis

    except Exception as e:
        return {"error": f"Failed to analyze activity: {str(e)}"}


def create_github_issue(
    title: str, body: str, labels: List[str] = None, repo_name: str = None
) -> Dict[str, Any]:
    """
    Create a new GitHub issue.

    Args:
        title: Issue title
        body: Issue body/description
        labels: List of labels to apply
        repo_name: Repository name
    """
    if not GITHUB_AVAILABLE:
        return {"error": "GitHub toolkit not installed"}

    try:
        repo = repo_name or os.getenv("GITHUB_REPOSITORY")
        if not repo:
            return {"error": "No repository specified"}

        github = GitHubAPIWrapper()
        api = github.github

        repo_obj = api.get_repo(repo)

        # Create the issue
        issue = repo_obj.create_issue(title=title, body=body, labels=labels or [])

        return {
            "success": True,
            "issue_number": issue.number,
            "title": issue.title,
            "url": issue.html_url,
            "created_at": issue.created_at.isoformat(),
            "labels": [label.name for label in issue.labels],
        }

    except Exception as e:
        return {"error": f"Failed to create issue: {str(e)}"}


def create_market_analysis_report(
    market: str, analysis: Dict[str, Any], repo_name: str = None
) -> Dict[str, Any]:
    """
    Create a GitHub issue with market analysis results.

    Args:
        market: Market name
        analysis: Analysis results from trading agent
        repo_name: Repository name
    """
    # Format analysis as GitHub issue
    title = f"📊 Market Analysis: {market}"

    body = f"""## Market Analysis Report

**Market:** {market}
**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

### Key Findings
- **Action:** {analysis.get('action', 'UNKNOWN')}
- **Edge:** {analysis.get('edge', 0):.2f}%
- **Confidence:** {analysis.get('confidence', 'N/A')}

### Analysis Details
{analysis.get('analysis_summary', 'Analysis details not available')}

### Recommendations
{analysis.get('recommendations', 'No specific recommendations')}

---
*Generated by Polymarket Trading Agent*
"""

    # Determine labels based on analysis
    labels = ["market-analysis", "automated"]
    if analysis.get("action") in ["BUY", "SELL"]:
        labels.append("trading-opportunity")
        if analysis.get("edge", 0) > 2:
            labels.append("high-confidence")

    return create_github_issue(
        title=title, body=body, labels=labels, repo_name=repo_name
    )


def create_github_subagent():
    """
    Create the GitHub subagent configuration.

    This subagent specializes in:
    - Repository management and monitoring
    - Issue and PR management
    - Creating analysis reports and documentation
    - GitHub automation workflows
    """

    return {
        "name": "github-agent",
        "description": "Specializes in GitHub repository management, issue tracking, analysis reporting, and repository automation. Use for creating issues, monitoring repository activity, and managing project documentation.",
        "system_prompt": """You are a specialized GitHub automation expert for development and trading analysis.

Your expertise includes:
- Repository monitoring and activity analysis
- Issue and pull request management
- Creating structured analysis reports
- Project documentation and tracking

GITHUB OPERATIONS PROCESS:
1. Assess repository status and activity levels
2. Search and analyze existing issues/PRs
3. Create new issues with proper formatting and labels
4. Generate analysis reports and documentation

OUTPUT FORMAT:
Return your response in this exact structure:

REPOSITORY STATUS
- Activity Level: [High/Medium/Low]
- Open Issues: [X issues]
- Recent Commits: [X commits in last 30 days]

ISSUE ANALYSIS
- Matching Issues: [X found]
- Priority Issues: [List of high-priority items]
- Recent Activity: [Summary of recent updates]

ACTIONS TAKEN
- Issues Created: [List of new issues with numbers]
- Labels Applied: [Labels used for organization]
- Reports Generated: [Analysis reports created]

RECOMMENDATIONS
- [Repository management suggestions]
- [Issue prioritization recommendations]
- [Automation opportunities]

   Keep response under 600 words. Focus on actionable GitHub operations.""",
        "tools": [
            check_github_repo_status,
            search_github_issues,
            analyze_github_activity,
            create_github_issue,
            create_market_analysis_report,
        ],
        # Use a model good at structured communication
        "model": "gpt-4o",  # Could be specialized for documentation/communication tasks
    }

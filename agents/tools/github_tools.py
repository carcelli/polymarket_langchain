"""
GitHub Tools for LangChain Agents

This module wraps the GitHub Toolkit to provide tools for interacting with GitHub repositories.
"""

import os
from typing import List, Optional

from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
from langchain_community.utilities.github import GitHubAPIWrapper

from agents.tooling import wrap_tool

def _get_github_toolkit() -> Optional[GitHubToolkit]:
    """Initialize and return the GitHubToolkit."""
    try:
        # Check for required environment variables
        if not all(os.getenv(var) for var in ["GITHUB_APP_ID", "GITHUB_APP_PRIVATE_KEY", "GITHUB_REPOSITORY"]):
             # Return None if env vars are missing, allowing the tools to handle it or fail gracefully
             return None

        github = GitHubAPIWrapper()
        toolkit = GitHubToolkit.from_github_api_wrapper(github)
        return toolkit
    except Exception as e:
        print(f"Error initializing GitHubToolkit: {e}")
        return None

def _get_issues_impl() -> str:
    """Fetch issues from the repository."""
    toolkit = _get_github_toolkit()
    if not toolkit:
        return "Error: GitHubToolkit not initialized. Check environment variables."
    
    # Find the 'Get Issues' tool
    for tool in toolkit.get_tools():
        if tool.name == "Get Issues":
             # The tool might expect arguments, but for simple fetching usually it doesn't or has defaults.
             # However, LangChain tools usually return structured output. 
             # We might need to invoke it.
             try:
                 return tool.invoke({})
             except Exception as e:
                 return f"Error invoking Get Issues: {e}"
    return "Error: Get Issues tool not found."

def _get_issue_impl(issue_number: int) -> str:
    """Fetch details about a specific issue."""
    toolkit = _get_github_toolkit()
    if not toolkit:
        return "Error: GitHubToolkit not initialized."

    for tool in toolkit.get_tools():
        if tool.name == "Get Issue":
            try:
                return tool.invoke({"issue_number": issue_number})
            except Exception as e:
                return f"Error invoking Get Issue: {e}"
    return "Error: Get Issue tool not found."

def _create_issue_comment_impl(issue_number: int, body: str) -> str:
    """Comment on a specific issue."""
    toolkit = _get_github_toolkit()
    if not toolkit:
        return "Error: GitHubToolkit not initialized."
        
    for tool in toolkit.get_tools():
        if tool.name == "Comment on Issue":
            try:
                return tool.invoke({"issue_number": issue_number, "body": body})
            except Exception as e:
                return f"Error invoking Comment on Issue: {e}"
    return "Error: Comment on Issue tool not found."

# Wrap the functions as tools
get_issues = wrap_tool(_get_issues_impl, name="get_issues", description="Fetch issues from the repository.")
get_issue = wrap_tool(_get_issue_impl, name="get_issue", description="Fetch details about a specific issue.")
create_issue_comment = wrap_tool(_create_issue_comment_impl, name="create_issue_comment", description="Comment on a specific issue.")

def get_github_tools() -> List:
    """Return a list of available GitHub tools."""
    return [
        get_issues,
        get_issue,
        create_issue_comment,
    ]

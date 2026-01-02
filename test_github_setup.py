#!/usr/bin/env python3
"""
Test GitHub App Setup

Run this script after updating your .env file with the real GitHub App private key.
"""

from dotenv import load_dotenv
import os
load_dotenv()

def test_github_setup():
    """Test if GitHub App setup is working."""
    print("🧪 Testing GitHub App Setup")
    print("=" * 40)

    # Check environment variables
    app_id = os.getenv('GITHUB_APP_ID')
    private_key = os.getenv('GITHUB_APP_PRIVATE_KEY')
    repository = os.getenv('GITHUB_REPOSITORY')

    print("📋 Configuration Check:")
    checks = [
        ("GITHUB_APP_ID", app_id, app_id is not None),
        ("GITHUB_APP_PRIVATE_KEY", "Set" if private_key else None, private_key is not None),
        ("GITHUB_REPOSITORY", repository, repository is not None),
    ]

    all_good = True
    for name, value, is_set in checks:
        status = "✅" if is_set else "❌"
        display_value = value if is_set else "Missing"
        print(f"  {status} {name}: {display_value}")
        if not is_set:
            all_good = False

    if not all_good:
        print("\n❌ Configuration incomplete. Please check your .env file.")
        return False

    # Check private key - could be file path or direct PEM content
    actual_key_path = private_key

    # Handle the case where path starts with /polymarket_langchain/
    if private_key.startswith('/polymarket_langchain/'):
        actual_key_path = private_key.replace('/polymarket_langchain/', './')

    if private_key.startswith('-----BEGIN'):
        print("\n✅ Configuration looks good")
        print("🔐 Private key is in PEM format (direct content)")
    elif os.path.exists(actual_key_path):
        print("\n✅ Configuration looks good")
        print(f"🔑 Private key file exists: {actual_key_path}")
        # Read file content for testing
        with open(actual_key_path, 'r') as f:
            file_content = f.read()
        if file_content.startswith('-----BEGIN'):
            print("   ✅ File contains valid PEM content")
        else:
            print("   ❌ File does not contain valid PEM content")
            return False
    else:
        print("\n❌ Private key is not valid PEM content and file doesn't exist")
        print("   Expected: -----BEGIN RSA PRIVATE KEY----- or valid file path")
        print(f"   Tried: {actual_key_path}")
        return False
    # Test actual GitHub connection
    print("\n🔗 Testing GitHub API connection...")
    try:
        from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
        from langchain_community.utilities.github import GitHubAPIWrapper

        # Use resolved path if needed
        key_to_use = actual_key_path if os.path.exists(actual_key_path) else private_key
        github = GitHubAPIWrapper(github_app_private_key=key_to_use)
        toolkit = GitHubToolkit.from_github_api_wrapper(github)
        tools = toolkit.get_tools()

        print(f"✅ GitHub Toolkit initialized with {len(tools)} tools:")
        for tool in tools:
            print(f"   • {tool.name}")

        # Test getting issues (safe read-only operation)
        print("\n📋 Testing issue retrieval...")
        get_issues_tool = next(t for t in tools if t.name == "Get Issues")
        issues = get_issues_tool.invoke({})

        print(f"✅ Success! Retrieved {len(issues)} issues from {repository}")

        if len(issues) > 0:
            print("📝 Sample issues:")
            for i, issue in enumerate(issues[:3]):
                if isinstance(issue, dict):
                    title = issue.get('title', 'No title')[:50]
                    number = issue.get('number', '?')
                    state = issue.get('state', 'unknown')
                else:
                    # Handle string format
                    title = str(issue)[:50]
                    number = '?'
                    state = 'unknown'
                print(f"   {i+1}. #{number} - {title}... ({state})")

        print("\n🎉 GitHub integration is ready!")
        print("🚀 You can now use GitHub automation in your agents!")
        return True

    except Exception as e:
        print(f"\n❌ GitHub API test failed: {e}")
        print("💡 Common issues:")
        print("   • Private key format is incorrect")
        print("   • GitHub App not installed on repository")
        print("   • App permissions insufficient")
        print("   • Repository path incorrect")
        return False

if __name__ == "__main__":
    success = test_github_setup()
    exit(0 if success else 1)

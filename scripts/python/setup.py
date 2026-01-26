import os
from dotenv import load_dotenv

load_dotenv()


# GitHub Integration Setup
def check_github_config():
    """Check if GitHub App configuration is properly set up."""
    required_vars = ["GITHUB_APP_ID", "GITHUB_APP_PRIVATE_KEY", "GITHUB_REPOSITORY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print(f"❌ Missing GitHub configuration: {', '.join(missing_vars)}")
        print("Please set these in your .env file:")
        print("  GITHUB_APP_ID=your_app_id")
        print("  GITHUB_APP_PRIVATE_KEY=/path/to/private_key.pem")
        print("  GITHUB_REPOSITORY=username/repo-name")
        return False

    # Check if private key looks valid
    private_key = os.getenv("GITHUB_APP_PRIVATE_KEY")

    # Handle file paths
    actual_key_path = private_key
    if private_key.startswith("/polymarket_langchain/"):
        actual_key_path = private_key.replace("/polymarket_langchain/", "./")

    if private_key.startswith("-----BEGIN"):
        print("   Private key: PEM format (direct content)")
    elif os.path.exists(actual_key_path):
        print(f"   Private key: File path ({actual_key_path})")
        # Check file content
        with open(actual_key_path, "r") as f:
            file_content = f.read()
        if not file_content.startswith("-----BEGIN"):
            print("⚠️  Private key file doesn't contain valid PEM content")
            return False
    else:
        print("⚠️  Private key doesn't look like PEM format (might be placeholder)")
        print("   Make sure you've replaced the placeholder with actual PEM content")
        return False

    print("✅ GitHub configuration loaded successfully!")
    print(f"   Repository: {os.getenv('GITHUB_REPOSITORY')}")
    print(f"   Branch: {os.getenv('GITHUB_BRANCH', 'main')}")
    print("   Private key: PEM format detected")

    # Test GitHub connection
    print("🔗 Testing GitHub API connection...")
    try:
        from langchain_community.agent_toolkits.github.toolkit import GitHubToolkit
        from langchain_community.utilities.github import GitHubAPIWrapper

        github = GitHubAPIWrapper(github_app_private_key=actual_key_path)
        toolkit = GitHubToolkit.from_github_api_wrapper(github)
        tools = toolkit.get_tools()

        # Quick test - get repo info
        get_issues_tool = next(t for t in tools if t.name == "Get Issues")
        issues = get_issues_tool.invoke({})
        print(f"   ✅ API connection successful! Found {len(issues)} issues")

        return True

    except Exception as e:
        print(f"   ❌ API connection failed: {str(e)[:100]}...")
        print("   Check: App installed on repo, permissions correct, private key valid")
        return False


# Polymarket Configuration
def check_polymarket_config():
    """Check if Polymarket API configuration is set up."""
    api_vars = ["OPENAI_API_KEY"]
    optional_vars = ["POLYGON_WALLET_PRIVATE_KEY", "NEWSAPI_API_KEY", "TAVILY_API_KEY"]

    missing_required = [var for var in api_vars if not os.getenv(var)]
    missing_optional = [var for var in optional_vars if not os.getenv(var)]

    if missing_required:
        print(f"❌ Missing required API keys: {', '.join(missing_required)}")
        return False

    print("✅ Polymarket configuration:")
    print(f"   LLM: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")
    print(f"   Trading: {'✅' if os.getenv('POLYGON_WALLET_PRIVATE_KEY') else '❌'}")
    print(f"   News: {'✅' if os.getenv('NEWSAPI_API_KEY') else '❌'}")
    print(f"   Research: {'✅' if os.getenv('TAVILY_API_KEY') else '❌'}")

    if missing_optional:
        print(f"ℹ️  Optional APIs available: {', '.join(missing_optional)}")

    return True


if __name__ == "__main__":
    print("🚀 Polymarket Agent Setup Check")
    print("=" * 40)

    # Check configurations
    polymarket_ok = check_polymarket_config()
    print()
    github_ok = check_github_config()

    print()
    if polymarket_ok and github_ok:
        print("🎉 All configurations loaded successfully!")
    else:
        print("⚠️  Some configurations are missing. Please check your .env file.")

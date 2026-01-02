import os
import sys
from pathlib import Path
import dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

dotenv.load_dotenv()

def test_github_env():
    required_vars = [
        "GITHUB_APP_ID",
        "GITHUB_APP_PRIVATE_KEY",
        "GITHUB_REPOSITORY",
        "OPENAI_API_KEY"
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
            
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please add them to your .env file.")
        return False
    
    print("✅ All required environment variables are present.")
    return True

def test_toolkit_initialization():
    print("\nAttempting to initialize GitHubToolkit...")
    from agents.tools.github_tools import _get_github_toolkit
    
    toolkit = _get_github_toolkit()
    if toolkit:
        print("✅ Successfully initialized GitHubToolkit.")
        print(f"Available tools: {[tool.name for tool in toolkit.get_tools()]}")
        return True
    else:
        print("❌ Failed to initialize GitHubToolkit.")
        return False

if __name__ == "__main__":
    print("--- GitHub Tools Integration Test ---")
    if test_github_env():
        test_toolkit_initialization()

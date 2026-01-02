import os
import sys
import json
import time
import shutil
import subprocess
import shlex
import httpx
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

import dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.table import Table
from rich.syntax import Syntax
from rich.prompt import Prompt, Confirm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

dotenv.load_dotenv()

console = Console()

class MemoryManager:
    """Manages persistent markdown memories for agents."""
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.base_dir = Path.home() / ".deepagents" / agent_name / "memories"
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_memory(self, topic: str, content: str):
        file_path = self.base_dir / f"{topic.lower().replace(' ', '_')}.md"
        with open(file_path, "w") as f:
            f.write(f"# Memory: {topic}\n\nLast updated: {datetime.now().isoformat()}\n\n{content}")
    
    def list_memories(self) -> List[str]:
        return [f.stem for f in self.base_dir.glob("*.md")]
    
    def read_memory(self, topic: str) -> str:
        file_path = self.base_dir / f"{topic}.md"
        if file_path.exists():
            return file_path.read_text()
        return ""
    
    def reset_memory(self):
        if self.base_dir.exists():
            shutil.rmtree(self.base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
            
    def copy_memory_from(self, source_agent: str):
        source_dir = Path.home() / ".deepagents" / source_agent / "memories"
        if source_dir.exists():
            for item in source_dir.glob("*.md"):
                shutil.copy2(item, self.base_dir)

    def get_all_context(self) -> str:
        context = []
        for file in self.base_dir.glob("*.md"):
            context.append(file.read_text())
        return "\n\n---\n\n".join(context)

class TodoManager:
    """Simple in-memory todo list for task planning."""
    def __init__(self):
        self.todos = []

    def manage_todos(self, action: str, task: str = None, index: int = None) -> str:
        if action == "add":
            if not task: return "Error: Task description required for 'add'"
            self.todos.append({"task": task, "status": "pending"})
            return f"Added task: {task}"
        elif action == "list":
            if not self.todos: return "No tasks in todo list."
            return "\n".join([f"{i}. [{'x' if t['status']=='done' else ' '}] {t['task']}" for i, t in enumerate(self.todos)])
        elif action == "complete":
            if index is None or index >= len(self.todos): return "Error: Invalid index"
            self.todos[index]['status'] = 'done'
            return f"Marked task {index} as done."
        return "Unknown action"

class Toolset:
    """Built-in capabilities for the Deep Agent."""
    
    @staticmethod
    def read_file(path: str) -> str:
        try:
            return Path(path).read_text()
        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    def write_file(path: str, content: str) -> str:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    @staticmethod
    def execute_shell(command: str) -> str:
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30
            )
            output = result.stdout
            if result.stderr:
                output += f"\nErrors:\n{result.stderr}"
            return output or "(No output)"
        except subprocess.TimeoutExpired:
            return "Error: Command timed out after 30 seconds"
        except Exception as e:
            return f"Error executing command: {e}"

    @staticmethod
    def web_search(query: str) -> str:
        from tavily import TavilyClient
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY not set"
        try:
            client = TavilyClient(api_key=api_key)
            results = client.search(query=query)
            return json.dumps(results, indent=2)
        except Exception as e:
            return f"Search error: {e}"
            
    @staticmethod
    def http_request(url: str, method: str = "GET", headers: Dict = None, data: Dict = None) -> str:
        try:
            with httpx.Client(timeout=10) as client:
                response = client.request(method, url, headers=headers, json=data)
                return f"Status: {response.status_code}\nBody: {response.text[:1000]}" # Truncate large responses
        except Exception as e:
            return f"HTTP Request Error: {e}"

from agents.utils.context import ContextManager, RuntimeContext

class DeepAgentCLI:
    def __init__(self, agent_name: str = "default", auto_approve: bool = False, sandbox_type: str = None):
        self.agent_name = agent_name
        self.auto_approve = auto_approve
        self.sandbox_type = sandbox_type
        
        # Initialize Context Manager
        runtime = RuntimeContext(
            user_id=os.getenv("USER", "default_user"),
            user_role="admin" if agent_name == "admin" else "trader",
            deployment_env="production" if sandbox_type else "development"
        )
        self.context_manager = ContextManager(runtime=runtime)
        self.memory = self.context_manager.store # Link to Store
        
        self.todos = TodoManager()
        self.history = []
        self.console = console
        self.total_tokens = 0
        
        # Base tools
        self.tools = [
            # ... (rest of the tools list stays same)
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read content of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative path to file"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative path to file"},
                            "content": {"type": "string", "description": "Content to write"}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "execute_shell",
                    "description": "Execute a bash command in the terminal",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Bash command to run"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web for up-to-date information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "http_request",
                    "description": "Make an HTTP request to an external API",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "URL to request"},
                            "method": {"type": "string", "description": "HTTP method (GET, POST, etc)", "default": "GET"},
                            "headers": {"type": "object", "description": "JSON headers"},
                            "data": {"type": "object", "description": "JSON body data"}
                        },
                        "required": ["url"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "manage_todos",
                    "description": "Manage task list (add, list, complete)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["add", "list", "complete"]},
                            "task": {"type": "string", "description": "Task description (for add)"},
                            "index": {"type": "integer", "description": "Task index (for complete)"}
                        },
                        "required": ["action"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "save_memory",
                    "description": "Save important information to long-term memory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "Memory topic/filename"},
                            "content": {"type": "string", "description": "Information to remember"}
                        },
                        "required": ["topic", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_issues",
                    "description": "Fetch issues from the repository.",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_issue",
                    "description": "Fetch details about a specific issue.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "issue_number": {"type": "integer", "description": "The issue number"}
                        },
                        "required": ["issue_number"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_issue_comment",
                    "description": "Comment on a specific issue.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "issue_number": {"type": "integer", "description": "The issue number"},
                            "body": {"type": "string", "description": "The comment body"}
                        },
                        "required": ["issue_number", "body"]
                    }
                }
            }
        ]

    def _call_llm(self, messages):
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        # Approximate usage tracking
        if response.usage:
            self.total_tokens += response.usage.total_tokens
        return response

    def _handle_tool_call(self, tool_call):
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        
        # Approval flow
        if not self.auto_approve:
            self.console.print(Panel(f"[bold yellow]Tool Call:[/bold yellow] {name}\n[dim]Args: {args}[/dim]", title="Approval Required"))
            if not Confirm.ask("Approve this action?"):
                return "Error: Action rejected by user"
        
        # Sandbox notification
        if self.sandbox_type and name in ["execute_shell", "write_file", "read_file"]:
             self.console.print(f"[dim](Executing in {self.sandbox_type} sandbox...)[/dim]")

        if name == "read_file":
            return Toolset.read_file(args["path"])
        elif name == "write_file":
            return Toolset.write_file(args["path"], args["content"])
        elif name == "execute_shell":
            return Toolset.execute_shell(args["command"])
        elif name == "web_search":
            return Toolset.web_search(args["query"])
        elif name == "http_request":
            return Toolset.http_request(args["url"], args.get("method", "GET"), args.get("headers"), args.get("data"))
        elif name == "manage_todos":
            return self.todos.manage_todos(args["action"], args.get("task"), args.get("index"))
        elif name == "save_memory":
            self.memory.save_memory(args["topic"], args["content"])
            return f"Memory saved: {args['topic']}"
        elif name == "get_issues":
            from agents.tools.github_tools import _get_issues_impl
            return _get_issues_impl()
        elif name == "get_issue":
            from agents.tools.github_tools import _get_issue_impl
            return _get_issue_impl(args["issue_number"])
        elif name == "create_issue_comment":
            from agents.tools.github_tools import _create_issue_comment_impl
            return _create_issue_comment_impl(args["issue_number"], args["body"])
        return "Unknown tool"

    def chat_loop(self):
        sandbox_info = f" (Sandbox: {self.sandbox_type})" if self.sandbox_type else ""
        self.console.print(Panel(f"[bold cyan]Deep Agents CLI[/bold cyan]{sandbox_info}\nInteractive terminal for building with persistent agents.", subtitle=f"Agent: {self.agent_name}"))
        
        while True:
            try:
                user_input = Prompt.ask(f"\n[bold green]{self.agent_name}[/bold green] >")
                
                if not user_input:
                    continue
                
                if user_input.startswith("/"):
                    self._handle_slash_command(user_input)
                    continue
                
                if user_input.startswith("!"):
                    self.console.print(Toolset.execute_shell(user_input[1:]))
                    continue

                self._process_task(user_input)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.console.print(f"[bold red]Error:[/bold red] {e}")

    def _handle_slash_command(self, cmd):
        if cmd == "/exit":
            sys.exit(0)
        elif cmd == "/clear":
            self.history = []
            self.console.print("[dim]Conversation history cleared.[/dim]")
        elif cmd == "/memories":
            memories = self.memory.list_memories()
            self.console.print(f"[bold]Memories:[/bold] {', '.join(memories) if memories else 'None'}")
        elif cmd == "/tokens":
            self.console.print(f"[yellow]Total Tokens Used (Session): {self.total_tokens}[/yellow]")
        else:
            self.console.print(f"[red]Unknown command: {cmd}[/red]")

    def _process_task(self, query):
        self.context_manager.update_state({"messages_count": len(self.history) // 2})
        context_block = self.context_manager.get_model_context()
        todos = self.todos.manage_todos("list")
        
        system_prompt = f"""You are a Deep Agent in an interactive CLI. 

{context_block}

Current Todo List:
{todos}

You have tools to read/write files, execute shell commands, make HTTP requests, search the web, and interact with GitHub.
Use 'manage_todos' to break down complex tasks.
Use 'save_memory' to remember important conventions or facts.
Be concise and helpful."""

        messages = [
            {"role": "system", "content": system_prompt},
            *self.history,
            {"role": "user", "content": query}
        ]

        # Use Live to show status
        with Live(self.console.print("[dim]Thinking...[/dim]"), refresh_per_second=4, console=self.console) as live:
            while True:
                response = self._call_llm(messages)
                msg = response.choices[0].message
                messages.append(msg)
                
                if msg.content:
                    self.console.print(Markdown(msg.content))
                
                if not msg.tool_calls:
                    break
                
                for tool_call in msg.tool_calls:
                    result = self._handle_tool_call(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": result
                    })
                    display_result = result[:500] + "..." if len(result) > 500 else result
                    self.console.print(Panel(f"[bold blue]Observation ({tool_call.function.name}):[/bold blue]\n{display_result}", title="Tool Output"))

        # Update persistent history (shortened)
        self.history = messages[-10:]

def list_agents():
    base_dir = Path.home() / ".deepagents"
    if not base_dir.exists():
        print("No agents found.")
        return
    
    agents = [d.name for d in base_dir.iterdir() if d.is_dir()]
    print("Available Agents:")
    for agent in agents:
        print(f"  - {agent}")

def reset_agent(agent_name: str, target: str = None):
    mem = MemoryManager(agent_name)
    mem.reset_memory()
    print(f"Reset memory for agent '{agent_name}'.")
    
    if target:
        mem.copy_memory_from(target)
        print(f"Copied memory from '{target}' to '{agent_name}'.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Deep Agents CLI")
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command")
    
    # List command
    subparsers.add_parser("list", help="List all agents")
    
    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset agent memory")
    reset_parser.add_argument("--agent", required=True, help="Agent name to reset")
    reset_parser.add_argument("--target", help="Copy memory from this agent")

    # Main CLI args (optional)
    parser.add_argument("--agent", default="default", help="Agent name")
    parser.add_argument("--auto-approve", action="store_true", help="Skip tool confirmation")
    parser.add_argument("--sandbox", choices=["modal", "daytona", "runloop"], help="Execute in remote sandbox")
    parser.add_argument("--sandbox-id", help="Reuse existing sandbox ID")
    parser.add_argument("--sandbox-setup", help="Path to sandbox setup script")

    args = parser.parse_args()

    if args.command == "list":
        list_agents()
    elif args.command == "reset":
        reset_agent(args.agent, args.target)
    else:
        # Default behavior: run CLI
        cli = DeepAgentCLI(
            agent_name=args.agent, 
            auto_approve=args.auto_approve,
            sandbox_type=args.sandbox
        )
        cli.chat_loop()

if __name__ == "__main__":
    main()
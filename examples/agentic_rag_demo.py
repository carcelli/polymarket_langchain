import os
import sys
import requests
from typing import List
from markdownify import markdownify

# Add project root to path
sys.path.append(str(Path(__file__).parents[1] / "src"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from polymarket_agents.langchain.agent import create_polymarket_agent
from polymarket_agents.tools.research_tools import fetch_documentation

# Constants for the demo
LLMS_TXT = "https://langchain-ai.github.io/langgraph/llms.txt"

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate


def run_agentic_rag_demo():
    print("🚀 Initializing Agentic RAG Demo...")

    # 1. Pre-fetch llms.txt content
    try:
        print(f"📥 Fetching documentation index from {LLMS_TXT}...")
        llms_txt_content = requests.get(LLMS_TXT, timeout=10.0).text
    except Exception as e:
        print(f"❌ Error fetching llms.txt: {e}")
        llms_txt_content = "Could not fetch documentation index."

    # 2. Define the specialized prompt
    template = """You are an expert developer and technical assistant for LangGraph.
    
    Your primary role is to help users with questions about LangGraph.

    Instructions:
    1. If a user asks a question you're unsure about or that involves API usage, 
       behavior, or configuration, you MUST use the `fetch_documentation` tool.
    2. When citing documentation, summarize clearly and include context.
    3. You can access official documentation from the approved sources listed below.
    4. You MUST consult the documentation to get up to date information before answering.

    Approved Sources (from llms.txt):
    {llms_txt_content}

    TOOLS:
    ------

    You have access to the following tools:

    {tools}

    To use a tool, please use the following format:

    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you have a response for the user, or if you do not need to use a tool, you MUST use the format:

    ```
    Thought: Do I need to use a tool? No
    Final Answer: [your response here]
    ```

    Begin!

    Question: {input}
    {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template).partial(
        llms_txt_content=llms_txt_content
    )

    # 3. Create the agent
    print("🤖 Creating Agentic RAG agent...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [fetch_documentation]

    agent = create_react_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    # 4. Run a sample query
    query = "Write a short example of a langgraph agent using the prebuilt create_react_agent. The agent should be able to look up stock pricing information."

    print(f"\n💬 Query: {query}\n")

    try:
        result = executor.invoke({"input": query})
        print("\n✅ Agent Response:\n")
        print(result["output"])
    except Exception as e:
        print(f"❌ Agent execution failed: {e}")


if __name__ == "__main__":
    run_agentic_rag_demo()

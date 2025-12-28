import sys
import os

# Ensure we can find the agents module
# scripts/test_graph.py -> scripts/ -> root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from agents.graph.state import AgentState
from agents.tools.market_tools import fetch_active_markets

# --- Node Definitions ---

def observer_node(state: AgentState):
    """
    The 'Observer' node. It uses the tool to sense the market.
    """
    print("👀 Node: Observer is waking up...")
    
    # In a real agent, the LLM decides to call the tool. 
    # Here, we hardcode the tool call for the 'Layer 1' test.
    # invoke returns a JSON string or dict depending on the tool? 
    # langchain tools usually return str. but our tool returns List[Dict].
    # Let's check what tool.invoke returns.
    
    try:
        tool_output = fetch_active_markets.invoke({"limit": 2})
    except Exception as e:
        print(f"❌ Error invoking tool: {e}")
        return {"messages": [HumanMessage(content=f"Error: {e}")]}
    
    # We transform the dict output back into our Pydantic objects for the State
    # (Assuming we handle the serialization/deserialization logic here)
    print(f"✅ Node: Found {len(tool_output)} markets.")
    
    # Return the update to the state
    return {"messages": [HumanMessage(content=f"Observed {len(tool_output)} markets")]}

# --- Graph Construction ---

def build_graph():
    workflow = StateGraph(AgentState)

    # 1. Add Nodes
    workflow.add_node("observer", observer_node)

    # 2. Add Edges (Simple linear flow for now)
    workflow.set_entry_point("observer")
    workflow.add_edge("observer", END)

    # 3. Compile
    return workflow.compile()

if __name__ == "__main__":
    print("🚀 Initializing LangGraph System...")
    try:
        app = build_graph()
        
        # Invoke
        initial_state = {"messages": [], "markets": [], "forecast_prob": 0.0, "trade_decision": {}}
        result = app.invoke(initial_state)
        
        print("\n--- Final State ---")
        if result and "messages" in result and result["messages"]:
            print(result["messages"][-1].content)
        else:
            print("No messages in final state.")
            
    except Exception as e:
        print(f"💥 Graph execution failed: {e}")
        import traceback
        traceback.print_exc()

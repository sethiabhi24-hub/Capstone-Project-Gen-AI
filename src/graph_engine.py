from langgraph.graph import StateGraph, START, END
from src.state import CapstoneState
from src.agents.supervisor import supervisor_node
from src.agents.specialists import *

def compile_workflow():
    workflow = StateGraph(CapstoneState)

    workflow.add_node('Supervisor',supervisor_node)
    workflow.add_node('support_agent',support_agent_node)
    workflow.add_node('retrieval_agent',retrieval_agent_node)
    workflow.add_node('escalation_agent',escalation_agent_node)

    workflow.add_edge(START, 'Supervisor')

    workflow.add_conditional_edges(
        'Supervisor',
        router_function,
        {
            "support_agent": "support_agent",
            "retrieval_agent": "retrieval_agent",
            "escalation_agent": "escalation_agent",
            END: END
        }
    )
    
    workflow.add_edge("support_agent", END)
    workflow.add_edge("retrieval_agent", END)
    workflow.add_edge("escalation_agent", END)

    return workflow.compile()
    
def router_function(state: CapstoneState) -> str:

    destination = state.get("next_node", "support_agent")
    if destination == "__end__":
        return END
        
    return destination


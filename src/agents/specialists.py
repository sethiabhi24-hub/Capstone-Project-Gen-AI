from src.state import CapstoneState
from src.rag_pipeline import query_knowledge_base
import json

def support_agent_node(state: CapstoneState):
    client = state["client"]
    support_prompt = f"""
    You are a AI support your job is to Greet the user nicely, also reply on queries for general and simpler queries, status.
    Customer Context: {json.dumps(state.get('customer_profile', {}))}
    Do not invent Polices.
    """

    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=state["messages"],
        config={"system_instruction":support_prompt}
    )

    print(f"response of support : {response}")
    json_txt = response.text
    return {"messages": [json_txt], 'next_node': "finish"}


def retrieval_agent_node(state: CapstoneState):
    client = state["client"]
    search_formulation = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=f"Extract the core search terms for our policy database from this query: {state['messages'][-1]}"
    ).text

    retrieved_context = query_knowledge_base(search_formulation, client)
    retrieval_prompt = f"""
    You are a AI Support Expert your job is to answer the user's question using only the retrieved policies below for acurate response on queries.
    If the answer is not in the policies, say you do not know and suggest opening a ticket. 
 
    {retrieved_context} 
    """

    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=state['messages'],
        config={"system_instruction":retrieval_prompt}
    )

    print(f"response of retrieval : {response}")
    json_txt = response.text
    return {"messages": [json_txt], 'next_node': "finish"}

def escalation_agent_node(state: CapstoneState):
    client = state["client"]
    escalation_context = query_knowledge_base("escalation", client)
    escalation_prompt = f"""
    You are a AI Escalation agent your job is to help user if user sound angry, user wants to escalate to higher tier.

    {escalation_context}
    """

    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=state['messages'],
        config={"system_instruction":escalation_prompt}
    )

    print(f"response of escalation : {response}")
    json_txt = response.text
    return {"messages": [json_txt], 'next_node': "finish"}

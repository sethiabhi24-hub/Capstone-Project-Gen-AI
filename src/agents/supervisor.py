import json
from google import genai
from google.genai import types
from src.state import CapstoneState

def supervisor_node(state: CapstoneState):
    print("\n[Supervisor]: Analyzing requests for routing...")
    client = state["client"]
    supervisor_prompt = """
    You are a AI supervisor your job is to route user query to correct specialist on bases of nodes defined in options.

    options:
    'support_agent': for general and simpler queries, status and for greetings.
    'retrieval_agent': for strict queries, policies where acurate response is needed.
    'escalation_agent': if user sound angry, user wants to escalate to higher tier.
    'finish': if conversation ended.

    always return response in JSON with mandatory key 'next_department' which is any one from above options.
    """

    response = client.models.generate_content(
        model='gemini-3.1-flash-lite',
        contents=state['messages'],
        config={"system_instruction":supervisor_prompt}
    )

    print(f"response of supervisor : {response}")
    json_txt = response.text.strip()
    if json_txt.startswith("```"):
        lines = [line for line in json_txt.splitlines() if not line.strip().startswith("```")]
        clean_json_txt = "\n".join(lines).strip()
    else:
        clean_json_txt = json_txt

    next_department = json.loads(clean_json_txt).get('next_department', 'support_agent')

    return {'next_node': next_department}
    

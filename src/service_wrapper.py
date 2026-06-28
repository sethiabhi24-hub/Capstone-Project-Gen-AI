from src.graph_engine import compile_workflow
import os
from google import genai

app_workflow = compile_workflow()

def process_chat_request(user_id: str, user_message: str) -> str:

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    config = {"configurable": {"thread_id": user_id}}
    
    input_state = {
        "messages": [user_message],
        "client": client
    }    

    output_state = app_workflow.invoke(input_state, config=config)
    
    messages_history = output_state.get("messages", [])
    
    return messages_history[-1]

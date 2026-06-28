import os

os.system("pip install google-genai langgraph langgraph-checkpoint-sqlite langsmith qdrant-client pypdf python-docx pandas flask python-dotenv")

from dotenv import load_dotenv
from google import genai

# Load environment variables from your .env file
load_dotenv()

# Verify that critical API keys are present
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("❌ CRITICAL ERROR: GEMINI_API_KEY is missing from your .env file.")

from src.rag_pipeline import build_vector_database
from app import app

def main():
    print("==================================================")
    print("🚀 STARTING NOVATECH ENTERPRISE AI APPLICATION")
    print("==================================================\n")
    
    # 1. Initialize the Google GenAI Client
    print("[System]: Initializing Google GenAI Client...")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    # 2. Build the Knowledge Layer (RAG)
    print("\n[System]: Step 1/2 — Building Knowledge Base Layer...")
    try:
        build_vector_database(client)
        print("✅ Knowledge Base built and sync complete.")
    except Exception as e:
        print(f"❌ Failed to initialize RAG pipeline: {e}")
        print("Continuing to launch web app, but retrieval functions may fail.")

    # 3. Open the Chatbot UI Web Gateway
    print("\n[System]: Step 2/2 — Launching Flask Interface Gateway...")
    print("👉 Click the link to open your Chatbot: http://127.0.0.1:5000")
    print("==================================================")
    
    # Run the web server (production-ready apps turn off debug mode)
    app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    main()
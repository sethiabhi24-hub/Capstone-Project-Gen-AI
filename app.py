from flask import Flask, render_template, request, jsonify
from src.service_wrapper import process_chat_request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.json
    user_id = data.get("user_id", "anonymous")
    message = data.get("message", "")
    print(f"📥 [Flask Terminal]: Received message from {user_id}: {message}")
    
    # Process requests using the main engine gateway
    ai_reply = process_chat_request(user_id, message)
    print(f"📤 [Flask Terminal]: Sending response back: {ai_reply}")
    return jsonify({"reply": ai_reply})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

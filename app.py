import json
import os
import sys

print("Demarrage...", flush=True)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

try:
    from agent import run_agent
    import agent as agent_module
    print("Agent OK", flush=True)
except Exception as e:
    print(f"ERREUR agent : {e}", flush=True)
    sys.exit(1)

try:
    from memory import memory_is_relevant, collection_count
    print("Memory OK", flush=True)
except Exception as e:
    print(f"ERREUR memory : {e}", flush=True)
    sys.exit(1)

app = Flask(__name__, static_folder="static")
CORS(app)

print("Flask OK", flush=True)


def sync_history(messages):
    agent_module.conversation_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in messages
        if msg["role"] in ("user", "assistant")
    ][-agent_module.MAX_HISTORY:]


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    question = data.get("question", "").strip()
    history = data.get("history", [])

    if not question:
        return jsonify({"error": "Question vide"}), 400

    sync_history(history)
    use_memory = memory_is_relevant(question)
    response = run_agent(question)

    return jsonify({
        "answer": response,
        "source": "memoire" if use_memory else "arxiv_web"
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify({
        "articles": collection_count()
    })


# Lancer Flask — fonctionne avec Render ET en local
port = int(os.environ.get("PORT", 8000))
print(f"Lancement sur 0.0.0.0:{port}", flush=True)
app.run(host="0.0.0.0", port=port, debug=False)
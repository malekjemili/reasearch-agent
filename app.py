import json
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from agent import run_agent
from memory import memory_is_relevant, collection_count
import agent as agent_module

app = Flask(__name__, static_folder="static")
CORS(app)


def sync_history(messages):
    """Synchronise l'historique depuis le front vers agent.py."""
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

    # Détecter la source utilisée depuis les logs
    return jsonify({
        "answer": response,
        "source": "memoire" if use_memory else "arxiv_web"
    })



# Dans la route /api/stats
@app.route("/api/stats", methods=["GET"])
def stats():
    return jsonify({
        "articles": collection_count()
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
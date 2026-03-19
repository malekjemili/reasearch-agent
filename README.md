# Research Assistant Agent

An autonomous AI agent that searches scientific literature, memorizes results, and synthesizes answers in real time.

---

## Demo

Live : [https:/reasearch-agent.onrender.com](https://reasearch-agent.onrender.com)  
GitHub : https://github.com/malekjemili/reasearch-agent

---

## Overview

The agent receives a natural language question, decides which source to query (ArXiv, web, or both), retrieves relevant documents, stores them in a persistent vector database, and returns a synthesized answer in French with cited sources.

On subsequent questions, the agent first checks its memory before making external requests, reducing latency and API usage.

---

## Architecture

```
User question
      |
      v
Memory check (Pinecone)
      |
   relevant?
   /       \
  yes       no
  |          |
  |     Generate query variants (multi-query RAG)
  |          |
  |     Search ArXiv + Web (Tavily)
  |          |
  |     Save to Pinecone
  |          |
   \       /
    Rank by semantic similarity
          |
    Synthesize answer (Groq / Llama 3.3)
          |
    Return response with sources
```

---

## Features

- Autonomous agent with plan, search, memorize, and synthesize loop
- Hybrid search across ArXiv (scientific papers) and the web (Tavily)
- Multi-query RAG : generates 3 query variants per question to maximize recall
- Semantic ranking of retrieved documents using cosine similarity
- Persistent vector memory in the cloud (Pinecone)
- Conversational memory across turns (last 10 exchanges)
- REST API with Flask
- Custom HTML/CSS/JS frontend with real-time RAG pipeline visualization
- Containerized with Docker and deployed on Render

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.3 70B via Groq API |
| Vector database | Pinecone (cloud, persistent) |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Scientific search | ArXiv API |
| Web search | Tavily API |
| Backend | Flask + Flask-CORS |
| Frontend | HTML / CSS / JavaScript |
| Container | Docker |
| Hosting | Render |

---

## Project Structure

```
research-agent/
├── app.py            # Flask server and API routes
├── agent.py          # Agent logic : planning, tool calls, synthesis
├── memory.py         # Pinecone vector store, RAG multi-query, ranking
├── tools.py          # ArXiv and Tavily search functions
├── Dockerfile        # Container definition
├── requirements.txt  # Python dependencies
└── static/
    └── index.html    # Frontend interface
```

---

## How It Works

### 1. Memory check

Before any external call, the agent computes the cosine similarity between the question embedding and stored vectors in Pinecone. If the best match exceeds a threshold of 0.75, it uses cached results.

### 2. Multi-query RAG

If no relevant memory is found, the agent generates 3 reformulations of the original question using the LLM. Each variant is used to query ArXiv and/or Tavily independently, maximizing coverage of relevant documents.

### 3. Semantic ranking

All retrieved documents are embedded and ranked by cosine similarity to the original question. The top 5 are passed to the synthesis step.

### 4. Synthesis

The LLM receives the ranked documents, the conversation history (last 10 turns), and the original question. It produces a structured answer in French with inline citations.

---

## Setup

### Prerequisites

- Python 3.11+
- API keys for Groq, Pinecone, and Tavily

### Installation

```bash
git clone https://github.com/malekjemili/reasearch-agent.git
cd reasearch-agent
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file at the root :

```
GROQ_API_KEY=your_groq_key
PINECONE_API_KEY=your_pinecone_key
TAVILY_API_KEY=your_tavily_key
```

### Pinecone index configuration

```
Name       : research-agent
Dimensions : 384
Metric     : cosine
Cloud      : AWS
Region     : us-east-1
```

### Run locally

```bash
python app.py
```

Open `http://localhost:5000`

### Run with Docker

```bash
docker build -t research-agent .
docker run -p 5000:5000 \
  -e GROQ_API_KEY=your_key \
  -e PINECONE_API_KEY=your_key \
  -e TAVILY_API_KEY=your_key \
  research-agent
```

---

## API

### POST /api/chat

Send a question and receive a synthesized answer.

Request :
```json
{
  "question": "Quelles sont les avancees en NLP ?",
  "history": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Response :
```json
{
  "answer": "Voici une synthese...",
  "source": "arxiv_web"
}
```

### GET /api/stats

Returns memory statistics.

Response :
```json
{
  "articles": 42
}
```

---

## Deployment

The application is deployed on Render using Docker. On each push to the `master` branch, Render automatically rebuilds and redeploys the container.

Environment variables are configured in the Render dashboard and injected at runtime — no secrets are stored in the repository.

---

## License

MIT

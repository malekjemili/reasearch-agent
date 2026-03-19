import os
import json
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
from pinecone import Pinecone, ServerlessSpec
import numpy as np

load_dotenv()

# Modèle d'embedding
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Client Groq pour le multi-query
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Client Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Créer l'index si il n'existe pas
INDEX_NAME = "research-agent"
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"Index '{INDEX_NAME}' cree")

index = pc.Index(INDEX_NAME)
print(f"Pinecone connecte — index : {INDEX_NAME}")


# ─────────────────────────────────────────
# FONCTIONS DE BASE
# ─────────────────────────────────────────

def save_articles(query: str, articles: list):
    """Sauvegarde les articles dans Pinecone."""
    vectors = []
    for i, article in enumerate(articles):
        text = f"{article['title']}. {article.get('summary', '')}"
        vec = embedder.encode(text).tolist()

        # ID unique basé sur l'URL
        doc_id = article.get("url", f"{query}_{i}").replace("/", "_").replace(":", "")[:50]

        vectors.append({
            "id": doc_id,
            "values": vec,
            "metadata": {
                "title": article["title"],
                "url": article.get("url", ""),
                "summary": article.get("summary", "")[:500],
                "authors": ", ".join(article["authors"]) if isinstance(article.get("authors"), list) else article.get("authors", ""),
                "published": article.get("published", ""),
                "source": article.get("source", "arxiv"),
                "query": query
            }
        })

    if vectors:
        index.upsert(vectors=vectors)
        print(f"{len(vectors)} articles sauvegardes dans Pinecone")


def search_memory(question: str, n_results: int = 3) -> list:
    """Recherche simple dans Pinecone."""
    stats = index.describe_index_stats()
    if stats.total_vector_count == 0:
        return []

    q_vec = embedder.encode(question).tolist()
    results = index.query(
        vector=q_vec,
        top_k=n_results,
        include_metadata=True
    )

    articles = []
    for match in results.matches:
        meta = match.metadata
        articles.append({
            "title": meta.get("title", ""),
            "url": meta.get("url", ""),
            "summary": meta.get("summary", ""),
            "authors": meta.get("authors", ""),
            "published": meta.get("published", ""),
            "source": meta.get("source", "arxiv"),
            "score": round(match.score, 3)
        })

    return articles


def memory_is_relevant(question: str, threshold: float = 0.75) -> bool:
    """Vérifie si la mémoire contient des articles pertinents."""
    stats = index.describe_index_stats()
    if stats.total_vector_count == 0:
        return False

    q_vec = embedder.encode(question).tolist()
    results = index.query(
        vector=q_vec,
        top_k=1,
        include_metadata=True
    )

    if not results.matches:
        return False

    score = results.matches[0].score
    print(f"Score memoire : {score:.3f} (seuil : {threshold})")
    return score >= threshold


def collection_count() -> int:
    """Retourne le nombre d'articles en mémoire."""
    try:
        stats = index.describe_index_stats()
        return stats.total_vector_count
    except Exception:
        return 0


# ─────────────────────────────────────────
# RAG AVANCE — MULTI-QUERY
# ─────────────────────────────────────────

def generate_query_variants(question: str) -> list:
    """Génère 3 reformulations différentes de la question."""
    print("Generation des variantes...")

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": """Génère 3 reformulations différentes d'une question pour maximiser les résultats.
Réponds UNIQUEMENT avec ce JSON :
{"queries": ["reformulation 1 en anglais", "reformulation 2 en anglais", "reformulation 3 en anglais"]}"""
            },
            {"role": "user", "content": f"Question : {question}"}
        ]
    )

    raw = response.choices[0].message.content.strip()
    try:
        data = json.loads(raw)
        queries = data["queries"]
        print(f"Variantes : {queries}")
        return queries
    except Exception:
        return [question]


def deduplicate(articles: list) -> list:
    """Supprime les doublons par URL."""
    seen = set()
    unique = []
    for a in articles:
        url = a.get("url", "")
        if url not in seen:
            seen.add(url)
            unique.append(a)
    return unique


def rank_articles(question: str, articles: list) -> list:
    """Trie les articles par pertinence sémantique."""
    if not articles:
        return []

    texts = [f"{a['title']}. {a.get('summary', '')}" for a in articles]
    q_vec = embedder.encode([question])
    t_vecs = embedder.encode(texts)
    scores = cosine_similarity(q_vec, t_vecs)[0]

    for i, article in enumerate(articles):
        article["score"] = round(float(scores[i]), 3)

    ranked = sorted(articles, key=lambda x: x["score"], reverse=True)

    print("Articles classes :")
    for a in ranked:
        print(f"  [{a['score']}] {a['title'][:60]}...")

    return ranked


def search_memory_advanced(question: str, n_results: int = 5) -> list:
    """RAG avancé multi-query avec Pinecone."""
    stats = index.describe_index_stats()
    if stats.total_vector_count == 0:
        return []

    print("\n--- RAG multi-query (Pinecone) ---")

    variants = generate_query_variants(question)

    all_articles = []
    for variant in variants:
        print(f"Recherche : '{variant}'")
        results = search_memory(variant, n_results=3)
        all_articles.extend(results)

    unique = deduplicate(all_articles)
    print(f"{len(all_articles)} trouves → {len(unique)} uniques")

    ranked = rank_articles(question, unique)
    return ranked[:n_results]


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":
    print(f"Articles en memoire : {collection_count()}")

    # Test sauvegarde
    test_articles = [{
        "title": "Attention Is All You Need",
        "summary": "We propose the Transformer architecture based solely on attention mechanisms.",
        "url": "https://arxiv.org/abs/1706.03762",
        "authors": ["Vaswani", "Shazeer"],
        "published": "2017-06-12",
        "source": "arxiv"
    }]

    save_articles("transformer attention", test_articles)
    print(f"Apres sauvegarde : {collection_count()} articles")

    # Test recherche
    found = search_memory("how do transformers work ?")
    print(f"\nArticles trouves : {len(found)}")
    for a in found:
        print(f"  [{a['score']}] {a['title']}")

    # Test pertinence
    relevant = memory_is_relevant("transformer neural network")
    print(f"\nMemoire pertinente : {relevant}")
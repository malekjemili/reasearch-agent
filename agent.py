import json
import os
from groq import Groq
from dotenv import load_dotenv
from tools import search_arxiv, search_web, search_all
from memory import save_articles, search_memory_advanced, memory_is_relevant

load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Historique conversationnel
conversation_history = []
MAX_HISTORY = 10


def run_agent(user_question: str) -> str:
    global conversation_history

    print(f"\nAgent reflechit a : {user_question}\n")

    # ETAPE 1 — Verifier la memoire d'abord
    if memory_is_relevant(user_question):
        print("Articles pertinents trouves en memoire !")
        articles = search_memory_advanced(user_question)
        source = "memoire"
    else:
        step1 = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": """Tu es un assistant de recherche.
Analyse la question et reponds UNIQUEMENT avec ce JSON :
{
  "query": "requete en anglais",
  "source": "arxiv" ou "web" ou "all"
}
Regles :
- "arxiv" si la question est sur des articles scientifiques, papiers de recherche
- "web" si la question est sur des actualites, produits, entreprises, evenements recents
- "all" si la question necessite les deux
Ne dis rien d'autre. Juste le JSON."""
                },
                {"role": "user", "content": user_question}
            ]
        )

        raw = step1.choices[0].message.content.strip()
        print(f"Plan genere : {raw}")

        try:
            plan = json.loads(raw)
            query = plan.get("query", user_question)
            source_type = plan.get("source", "all")
        except Exception:
            query = user_question
            source_type = "all"

        # ETAPE 3 — Chercher selon la source choisie
        print(f"Source : {source_type} | Requete : {query}")

        if source_type == "arxiv":
            articles = search_arxiv(query, max_results=5)
            source = "ArXiv"
        elif source_type == "web":
            articles = search_web(query, max_results=5)
            source = "Web"
        else:
            articles = search_all(query, max_results=6)
            source = "ArXiv + Web"

        print(f"{len(articles)} resultats trouves depuis {source}")
        save_articles(query, articles)
        # ETAPE 4 — Sauvegarder en memoire
        save_articles(query, articles)
        source in  ["ArXiv", "web"] 

    # ETAPE 5 — Synthetiser avec historique conversationnel
    articles_text = json.dumps(articles, ensure_ascii=False, indent=2)

    # Construire les messages avec historique
    messages = [
        {
            "role": "system",
            "content": """Tu es un assistant de recherche scientifique expert.
Tu as acces a l'historique de la conversation pour maintenir le contexte.
Regles :
- Si l'utilisateur fait reference a une question precedente, utilise l'historique.
- Synthetise les articles en francais avec les sources.
- Si la question est un suivi, connecte la reponse aux echanges precedents.
- Sois concis et precis."""
        }
    ]

    # Ajouter l'historique recent
    messages.extend(conversation_history[-MAX_HISTORY:])

    # Ajouter la question actuelle avec les articles
    messages.append({
        "role": "user",
        "content": f"""Question : {user_question}
Source : {source}

Articles :
{articles_text}

Synthese en francais avec les points cles et les sources.
Si cette question est un suivi, fais le lien avec ce qui a ete dit precedemment."""
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages
    )

    answer = response.choices[0].message.content

    # Mettre a jour l'historique (question simple, sans les articles)
    conversation_history.append({"role": "user", "content": user_question})
    conversation_history.append({"role": "assistant", "content": answer})

    # Garder seulement les MAX_HISTORY derniers messages
    if len(conversation_history) > MAX_HISTORY:
        conversation_history = conversation_history[-MAX_HISTORY:]

    print(f"Historique : {len(conversation_history)} messages")

    return answer


if __name__ == "__main__":
    print("=== Test 1 ===")
    r1 = run_agent("Quelles sont les dernieres avancees en traitement du langage naturel ?")
    print(r1[:300])

    print("\n=== Test 2 - suivi ===")
    r2 = run_agent("Approfondis le premier article")
    print(r2[:300])

    print("\n=== Test 3 - suivi ===")
    r3 = run_agent("Quels sont les auteurs principaux dans ce domaine ?")
    print(r3[:300])
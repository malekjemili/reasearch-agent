import arxiv
import os
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()
def search_arxiv(query: str, max_results: int = 5) -> list:
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    results = []
    for paper in client.results(search):
        results.append({
            "title": paper.title,
            "summary": paper.summary[:500],
            "url": paper.entry_id,
            "authors": [a.name for a in paper.authors[:3]],
            "published": paper.published.strftime("%Y-%m-%d")  
        })
    return results
def search_web(query: str, max_results: int = 5) -> list:
    """Recherche sur tout le web via Tavily."""
    try:
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",  # recherche approfondie
            include_answer=True       # résumé automatique
        )
        results = []
        for r in response.get("results", []):
            results.append({
                "title": r.get("title", ""),
                "summary": r.get("content", "")[:500],
                "url": r.get("url", ""),
                "authors": [],
                "published": r.get("published_date", ""),
                "source": "web"
            })
        return results
    except Exception as e:
        print(f"Erreur Tavily : {e}")
        return []


def search_all(query: str, max_results: int = 5) -> list:
    """Cherche sur ArXiv ET sur le web, fusionne les résultats."""
    arxiv_results = search_arxiv(query, max_results=3)
    web_results = search_web(query, max_results=3)

    # Fusionner et dédupliquer
    all_results = arxiv_results + web_results
    seen_urls = set()
    unique = []
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique.append(r)

    return unique[:max_results]


if __name__ == "__main__":
    print("=== Test ArXiv ===")
    for a in search_arxiv("large language models"):
        print(f"[ArXiv] {a['title']}")

    print("\n=== Test Web ===")
    for a in search_web("latest AI research 2024"):
        print(f"[Web] {a['title']} — {a['url']}")

    print("\n=== Test Fusion ===")
    for a in search_all("NLP advances 2024"):
        print(f"{a['title']}")
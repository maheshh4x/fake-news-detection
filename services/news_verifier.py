from ddgs import DDGS


def fetch_related_articles(query: str, max_results=5):
    results = []

    if not query or len(query.strip()) < 5:
        return results

    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append({
                    "title": r.get("title", "No title"),
                    "link": r.get("url", ""),
                    "source": r.get("source", "Unknown")
                })
    except:
        return []

    return results

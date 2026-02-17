from langchain_core.tools import tool
from app.core.config import firecrawl


@tool
def web_search(query: str):
    """
    Search the web for information.
    """
    response = firecrawl.search(
        query,
        sources=["web"],
        limit=10,
    )

    if not response or not hasattr(response, "web"):
        return "No results found."

    formatted_results = []
    if response.web:
        for result in response.web:
            title = getattr(result, "title", "No Title")
            url = getattr(result, "url", "#")
            description = getattr(result, "description", "No description available.")
            formatted_results.append(f"### [{title}]({url})\n{description}\n")

    if not formatted_results:
        return "No results found."

    return "\n".join(formatted_results)

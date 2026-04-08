from fastmcp import FastMCP
from pydantic import Field
from sentence_transformers import SentenceTransformer

from config.settings import settings
from rag import zvec_db

mcp = FastMCP(" MPF Employer RAG")


@mcp.tool()
def query_rag(
    query: str = Field(description="The user question about MPF obligations"),
) -> str:
    """Query the MPF knowledge base and get an answer."""
    model = SentenceTransformer(settings.embedding_model)
    query_embedding = model.encode([query])[0].tolist()
    results = zvec_db.search(query_embedding, top_k=settings.search_top_k)

    if not results:
        return "No relevant information found."

    context = "\n\n".join(
        [r["text"].replace("<br>", "\n").replace("</p>", "\n\n")[:800] for r in results]
    )

    return f"Context:\n{context}\n\nQuery: {query}"


@mcp.tool()
def get_stats() -> dict:
    """Get knowledge base statistics."""
    return zvec_db.get_stats()


if __name__ == "__main__":
    print(f"Starting MCP server on {settings.mcp_host}:{settings.mcp_port}")
    mcp.run(transport="streamable-http", host=settings.mcp_host, port=settings.mcp_port)

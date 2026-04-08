from fastmcp import FastMCP
from pydantic import Field

from config.settings import settings


mcp = FastMCP("MPF Employer RAG")


@mcp.tool()
def query_rag(
    query: str = Field(description="The user question about MPF obligations"),
) -> str:
    """Query the MPF knowledge base and get an answer."""
    from agents.rag_tools import search_documents

    return search_documents.invoke(query)


@mcp.tool()
def get_stats() -> dict:
    """Get knowledge base statistics."""
    return zvec_db.get_stats()


if __name__ == "__main__":
    print(f"Starting MCP server on {settings.mcp_host}:{settings.mcp_port}")
    mcp.run(transport="streamable-http", host=settings.mcp_host, port=settings.mcp_port)

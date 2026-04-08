from sentence_transformers import SentenceTransformer
from pydantic import Field
from langchain.tools import tool

from config.settings import settings
from rag import zvec_db


class EmbeddingModel:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    def embed_single(self, text: str) -> list[float]:
        return self.embed([text])[0]


_embedding_model = EmbeddingModel()


@tool
def search_documents(query: str) -> str:
    """Search for relevant documents in the knowledge base.

    Use this tool to find information about MPF (Mandatory Provident Fund) obligations,
    employer responsibilities, employee enrolment, contributions, and related topics.

    Args:
        query: The search query in natural language.

    Returns:
        Formatted search results with relevant document excerpts.
    """
    query_embedding = _embedding_model.embed_single(query)
    results = zvec_db.search(query_embedding, top_k=settings.search_top_k)

    formatted = []
    for i, r in enumerate(results, 1):
        text = r["text"].replace("<br>", "\n").replace("</p>", "\n\n")[:500]
        formatted.append(f"--- Result {i} (score: {r['score']:.3f}) ---\n{text}\n")

    return "\n\n".join(formatted) if formatted else "No results found."


@tool
def get_knowledge_stats() -> str:
    """Get statistics about the knowledge base.

    Returns:
        Information about the number of indexed documents.
    """
    stats = zvec_db.get_stats()
    return f"Total indexed documents: {stats['total_documents']}"


def create_rag_tools() -> list:
    return [search_documents, get_knowledge_stats]

import zvec
from typing import Protocol, runtime_checkable
from config.settings import settings


@runtime_checkable
class VectorDB(Protocol):
    def insert(self, documents: list[dict]) -> None: ...
    def query(self, vector: list[float], top_k: int) -> list[dict]: ...
    def get_stats(self) -> dict: ...


def get_embedding_dim() -> int:
    return settings.embedding_dim


def create_collection() -> zvec.Collection:
    settings.zvec_path.parent.mkdir(parents=True, exist_ok=True)

    schema = zvec.CollectionSchema(
        name="documents",
        vectors=zvec.VectorSchema(
            "embedding",
            zvec.DataType.VECTOR_FP32,
            get_embedding_dim(),
        ),
        fields=zvec.FieldSchema(
            "text",
            zvec.DataType.STRING,
        ),
    )

    return zvec.create_and_open(
        path=str(settings.zvec_path),
        schema=schema,
    )


def open_collection() -> zvec.Collection:
    return zvec.open(str(settings.zvec_path))


def index_documents(docs: list[dict]) -> None:
    """Index documents with their embeddings."""
    collection = open_collection()

    documents = [
        zvec.Doc(
            id=doc["id"],
            vectors={"embedding": doc["embedding"]},
            fields={"text": doc.get("text", "")},
        )
        for doc in docs
    ]

    collection.insert(documents)


def search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """Search for similar documents."""
    collection = open_collection()

    results = collection.query(
        zvec.VectorQuery(
            field_name="embedding",
            vector=query_embedding,
        ),
        topk=top_k,
    )

    return [
        {
            "id": r.id,
            "score": r.score,
            "text": r.field("text") if r.has_field("text") else "",
        }
        for r in results
    ]


def get_stats() -> dict:
    """Get collection statistics."""
    collection = open_collection()
    return {
        "total_documents": collection.stats.doc_count,
    }

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from markdown import markdown
from pathlib import Path
import zvec

from config.settings import settings
from rag import zvec_db


def load_markdown_documents(path: Path) -> list[str]:
    """Load and split markdown documents into chunks."""
    content = path.read_text(encoding="utf-8")
    html = markdown(content)
    text = html.replace("<br>", "\n").replace("</p>", "\n\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.split_text(text)


def embed_documents(chunks: list[str]) -> list[dict]:
    """Embed document chunks."""
    model = SentenceTransformer(settings.embedding_model)
    embeddings = model.encode(chunks, show_progress_bar=True)

    return [
        {
            "id": f"doc_{i}",
            "text": chunk,
            "embedding": embedding.tolist(),
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]


def ensure_collection():
    """Create Zvec collection if it doesn't exist."""
    settings.zvec_path.parent.mkdir(parents=True, exist_ok=True)

    if settings.zvec_path.exists():
        return zvec.open(str(settings.zvec_path))

    schema = zvec.CollectionSchema(
        name="documents",
        vectors=zvec.VectorSchema(
            "embedding",
            zvec.DataType.VECTOR_FP32,
            settings.embedding_dim,
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


def main():
    print("Loading documents...")
    chunks = load_markdown_documents(settings.documents_path)
    print(f"Loaded {len(chunks)} chunks")

    print("Creating embedding model...")
    docs = embed_documents(chunks)
    print(f"Embedded {len(docs)} documents")

    print("Creating/indexing to Zvec...")
    ensure_collection()
    zvec_db.index_documents(docs)
    print("Done!")

    stats = zvec_db.get_stats()
    print(f"Total documents indexed: {stats['total_documents']}")


if __name__ == "__main__":
    main()

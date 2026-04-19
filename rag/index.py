"""Index documents into Qdrant with contextual retrieval + hybrid vectors."""
from __future__ import annotations

import os, sys, uuid
from pathlib import Path

import anthropic
from chonkie import SDPMChunker
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models
from unstructured.partition.auto import partition

COLLECTION = os.getenv("QDRANT_COLLECTION", "kb")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_KEY = os.getenv("QDRANT_API_KEY", "")
TENANT_ID = os.getenv("TENANT_ID", "default")

CONTEXT_PROMPT = """Documento completo:
<documento>
{document}
</documento>

Chunk a indexar:
<chunk>
{chunk}
</chunk>

Escribe 1-3 frases situándolo (tema, entidad, fecha). No repitas info del chunk."""


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION):
        return
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={"dense": models.VectorParams(size=1024, distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )


def contextualize(client: anthropic.Anthropic, doc: str, chunk: str) -> str:
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": CONTEXT_PROMPT.format(document=doc[:8000], chunk=chunk)}],
    )
    return resp.content[0].text.strip()


def index_path(path: Path) -> None:
    qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_KEY)
    ensure_collection(qc)

    anth = anthropic.Anthropic()
    chunker = SDPMChunker(embedding_model="BAAI/bge-m3", chunk_size=512, threshold=0.75)
    dense_model = TextEmbedding("BAAI/bge-m3")
    sparse_model = SparseTextEmbedding("Qdrant/bm25")

    for file in path.rglob("*"):
        if not file.is_file():
            continue
        text = "\n\n".join(str(el) for el in partition(str(file)))
        print(f"parsed {file.name} ({len(text)} chars)")
        points: list[models.PointStruct] = []
        for chunk in chunker.chunk(text):
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            ctx = contextualize(anth, text, chunk_text)
            full = f"{ctx}\n\n---\n\n{chunk_text}"
            dense = next(dense_model.embed([full]))
            sparse = next(sparse_model.embed([full]))
            points.append(models.PointStruct(
                id=uuid.uuid4().hex,
                vector={"dense": dense.tolist(), "bm25": sparse.as_object()},
                payload={
                    "text": chunk_text,
                    "context": ctx,
                    "source": str(file),
                    "tenant_id": TENANT_ID,
                },
            ))
            if len(points) >= 128:
                qc.upsert(collection_name=COLLECTION, points=points)
                points = []
        if points:
            qc.upsert(collection_name=COLLECTION, points=points)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m rag.index <path>")
        sys.exit(1)
    index_path(Path(sys.argv[1]))

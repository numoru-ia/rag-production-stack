> Read the full article on Numoru: https://numoru.com/en/contributions/context-engineering-rag-produccion

# rag-production-stack

Production RAG pipeline with Chonkie + Qdrant hybrid + BGE-reranker + Contextual Retrieval + RedisVL cache + Ragas evals.

Companion to: [Context engineering: por qué tu RAG se rompe a los 50k tokens](https://numoru.com/contribuciones/context-engineering-rag-produccion).

## Run

```bash
docker compose up -d
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m rag.index /path/to/docs/
python -m rag.query "tu pregunta"
```

## Pipeline

```
parse (Unstructured/Firecrawl)
  → chunk (Chonkie SDPM)
  → contextualize (Claude Haiku batch)
  → embed (BGE-m3 or OpenAI 3-small) + BM25
  → upsert (Qdrant hybrid collection)

query-time:
  cache lookup (RedisVL 0.92 similarity)
  → hybrid search top-40
  → BGE-reranker v2-m3 → top-6
  → LLM prompt
```

## Metrics

Typical improvement (from our client cases):

| Config | Recall@5 | Faithfulness |
|---|---|---|
| Naive | 0.62 | 0.71 |
| + SDPM chunking | 0.71 | 0.76 |
| + Contextual Retrieval | 0.78 | 0.83 |
| + Hybrid search | 0.85 | 0.85 |
| **+ BGE-reranker** | **0.91** | **0.90** |

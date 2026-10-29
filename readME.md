# ContractScan

Upload a contract PDF, split it into clauses, embed them and search them. The end goal is flagging risky clauses with an LLM that cites the exact contract text.

## How to run

From the project root (imports use `from src...`):

```
python -m venv venv
venv\Scripts\activate            # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
python tests/smoke_test.py
streamlit run main.py
```

The first run downloads the embedding model (all-MiniLM-L6-v2), so it needs internet.

## Pipeline so far

1. **Parse** (`src/parser.py`): reads the PDF line by line with PyMuPDF, merges text on the same visual row, removes running headers/footers from the page margins, detects top-level headings ("1. Title", "1 Title", "Clause 1", Schedule/Annex/Appendix) and groups lines into clauses and sub-clauses (6.1, 6.2...).
2. **Chunk** (`src/chunker.py`): one chunk per sub-clause, split further by sentence/word if it exceeds the embedding model's own token limit. The heading is prepended to the text that gets embedded; the raw text is kept for citation. Chunk IDs are `{doc_id}_0001`, where `doc_id` is a hash of the PDF.
3. **Embed** (`src/embedder.py`): sentence-transformers, normalised embeddings, warns if anything would be truncated.
4. **Store and search** (`src/vectordb.py`): ChromaDB with cosine distance; search is filtered by `doc_id` and returns `{id, text, metadata, score}`. An already-indexed PDF is not embedded again.

## Roadmap

- [x] Parsing, chunking, embedding, vector search
- [ ] Test on more contracts; build a labelled test set from CUAD
- [ ] Risk knowledge base (15–20 categories)
- [ ] Per-category retrieval + cross-encoder reranking
- [ ] LLM analysis with JSON schema, Pydantic validation and quote checking
- [ ] Evaluation: retrieval hit rate, citation validity
- [ ] FastAPI + Docker + deployment

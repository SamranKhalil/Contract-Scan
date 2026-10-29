from pathlib import Path

import chromadb

# Absolute path so the database is the same wherever the app is started from
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "chroma_db"


class VectorDB:

    def __init__(self, path=DEFAULT_PATH):

        self.client = chromadb.PersistentClient(path=str(path))

    def get_collection(self, name):

        collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

        return collection

    def add(self, collection, ids, texts, embeddings, metadatas):
        """
        Generic insert, used for contract chunks and the risk knowledge base.
        """

        collection.upsert(
            ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas
        )

    def add_chunks(self, collection, chunks, embeddings):

        ids = []
        texts = []
        metadatas = []

        for chunk in chunks:

            ids.append(chunk["chunk_id"])

            texts.append(chunk["text"])

            metadatas.append(
                {
                    "doc_id": chunk["doc_id"],
                    # convert None -> empty string
                    "clause_number": chunk["clause_number"] or "",
                    "sub_clause_number": chunk["sub_clause_number"] or "",
                    "heading": chunk["heading"],
                    "clause_type": chunk["clause_type"],
                    "page": chunk["page"],
                    "tokens": chunk["tokens"],
                }
            )

        self.add(collection, ids, texts, embeddings, metadatas)

    def has_document(self, collection, doc_id):

        result = collection.get(where={"doc_id": doc_id}, limit=1)

        return len(result["ids"]) > 0

    def delete_document(self, collection, doc_id):

        collection.delete(where={"doc_id": doc_id})

    def search(self, collection, query_embedding, top_k=5, doc_id=None):
        """
        Returns:
        [
            {id, text, metadata, score}   # score = cosine similarity
        ]
        """

        where = {"doc_id": doc_id} if doc_id else None

        results = collection.query(
            query_embeddings=[query_embedding], n_results=top_k, where=where
        )

        return [
            {
                "id": chunk_id,
                "text": text,
                "metadata": metadata,
                "score": 1 - distance,
            }
            for chunk_id, text, metadata, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            )
        ]

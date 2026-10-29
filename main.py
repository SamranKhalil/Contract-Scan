import hashlib

import streamlit as st

from src.chunker import create_chunks
from src.embedder import Embedder
from src.parser import parse_contract
from src.vectordb import VectorDB

COLLECTION_NAME = "contracts"


# Loaded once per server process instead of on every rerun
@st.cache_resource
def load_embedder():
    return Embedder()


@st.cache_resource
def load_vectordb():
    return VectorDB()


st.title("Contract Scan PDF Parser")
uploaded_file = st.file_uploader("Upload pdf legal document", type=["pdf"])

if uploaded_file:
    # getvalue() rather than read(): read() returns b"" on a Streamlit rerun
    pdf_bytes = uploaded_file.getvalue()

    # Same file -> same doc_id, so chunk IDs never collide across contracts
    doc_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]

    embedder = load_embedder()
    vectordb = load_vectordb()
    collection = vectordb.get_collection(COLLECTION_NAME)

    # parsing the pdf
    clauses = parse_contract(pdf_bytes)

    if not clauses:
        st.error("No text found in this PDF. Scanned PDFs are not supported yet.")
        st.stop()

    # chunking the pdf
    chunks = create_chunks(clauses, doc_id, embedder)
    st.success(f"Extracted {len(clauses)} clauses and {len(chunks)} chunks")

    # adding chunks in the db for future referencing
    if vectordb.has_document(collection, doc_id):
        st.info("This contract is already indexed; skipping embedding.")
    else:
        with st.spinner("Embedding chunks..."):
            embeddings = embedder.embed_texts([chunk["embed_text"] for chunk in chunks])
            vectordb.add_chunks(collection, chunks, embeddings)

    # Query
    st.header("Search this contract")
    query = st.text_input("Query", value="supplier liability for indirect damages")

    if query:
        query_embedding = embedder.embed_query(query)
        results = vectordb.search(collection, query_embedding, top_k=3, doc_id=doc_id)

        for result in results:
            metadata = result["metadata"]
            clause = metadata["sub_clause_number"] or metadata["clause_number"]

            st.markdown(
                f"**{metadata['heading']}** · clause {clause or '-'} · "
                f"page {metadata['page']} · score {result['score']:.2f}"
            )
            st.write(result["text"])
            st.divider()

    with st.expander(f"All chunks ({len(chunks)})"):
        for chunk in chunks:
            st.subheader(f"{chunk['heading']}")

            st.write(f"**Chunk ID:** {chunk['chunk_id']}")
            st.write(f"**Clause Number:** {chunk['sub_clause_number'] or chunk['clause_number']}")
            st.write(f"**Clause Type:** {chunk['clause_type']}")
            st.write(f"**Page:** {chunk['page']}")
            st.write(f"**Tokens:** {chunk['tokens']}")

            st.write("**Text:**")
            st.write(chunk["text"])

            st.divider()

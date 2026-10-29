"""
Smoke test for the parse -> chunk -> embed -> store -> search pipeline.

Run from the project root:
    python tests/smoke_test.py
"""

import hashlib
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chunker import create_chunks, infer_clause_type, split_sentences  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.parser import join_lines, parse_contract  # noqa: E402
from src.vectordb import VectorDB  # noqa: E402

PDF_PATH = ROOT / "test_contract_contractscan.pdf"

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail and not condition else ""))
    if not condition:
        failures.append(name)


# ---------------------------------------
# Unit checks that need no PDF or model
# ---------------------------------------

check("lines joined with a space", join_lines(["The Supplier", "shall pay"]) == "The Supplier shall pay")
check("hyphenated line break rejoined", join_lines(["this Agree-", "ment"]) == "this Agreement")
check("'Relationship' is not IP", infer_clause_type("Relationship of the Parties") != "Intellectual Property")
check("'IP' still detected", infer_clause_type("Background IP") == "Intellectual Property")
check("'Ltd.' does not split a sentence", len(split_sentences("NovaTech Ltd. is the Supplier. It agrees.")) == 2)

# ---------------------------------------
# Parser on the test contract
# ---------------------------------------

pdf_bytes = PDF_PATH.read_bytes()
doc_id = hashlib.sha256(pdf_bytes).hexdigest()[:16]

clauses = parse_contract(pdf_bytes)
headings = [c["heading"] for c in clauses]
full_text = " ".join(c["content"] for c in clauses)

check("11 numbered clauses found", sum(1 for c in clauses if (c["clause_number"] or "").isdigit()) == 11, headings)
check("schedules found", sum(h.upper().startswith("SCHEDULE") for h in headings) == 2, headings)
check("preamble captured", headings[0] == "PREAMBLE")

fused = re.findall(r"\b(?:Supplier|Client|Party|Agreement)(?:shall|may|will|and|the)\b", full_text)
check("no fused words", not fused, fused[:5])

liability = next(c for c in clauses if c["clause_number"] == "6")
check("clause 6 split into sub-clauses", [s["number"] for s in liability["sub_clauses"]] == ["6.1", "6.2", "6.3", "6.4"])

# ---------------------------------------
# Chunker + embedder
# ---------------------------------------

embedder = Embedder()
chunks = create_chunks(clauses, doc_id, embedder)
ids = [c["chunk_id"] for c in chunks]

check("chunk IDs unique", len(ids) == len(set(ids)))
check("chunk IDs prefixed with doc_id", all(i.startswith(doc_id + "_") for i in ids))
check(
    "no chunk exceeds the model limit",
    all(c["tokens"] + 2 <= embedder.max_tokens for c in chunks),
    max(c["tokens"] for c in chunks),
)
check("no empty chunks", all(c["text"].strip() for c in chunks))

# ---------------------------------------
# Vector DB (temporary folder so the real DB is untouched)
# ---------------------------------------

# Chroma keeps files open on Windows, so cleanup errors are ignored
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
    vectordb = VectorDB(path=tmp)
    collection = vectordb.get_collection("contracts")

    vectordb.add_chunks(collection, chunks, embedder.embed_texts([c["embed_text"] for c in chunks]))
    check("document is indexed", vectordb.has_document(collection, doc_id))

    # A second contract with the same structure must not overwrite the first
    other_chunks = create_chunks(clauses[:2], "otherdoc", embedder)
    vectordb.add_chunks(collection, other_chunks, embedder.embed_texts([c["embed_text"] for c in other_chunks]))
    check("second document does not overwrite first", collection.count() == len(chunks) + len(other_chunks))

    query = embedder.embed_query("supplier liability for indirect damages")
    results = vectordb.search(collection, query, top_k=3, doc_id=doc_id)

    check("search filtered by doc_id", all(r["metadata"]["doc_id"] == doc_id for r in results))
    check(
        "liability clause is the top hit",
        results[0]["metadata"]["clause_number"] == "6",
        [(r["metadata"]["sub_clause_number"], round(r["score"], 2)) for r in results],
    )

    vectordb.delete_document(collection, "otherdoc")
    check("delete_document removes only that doc", collection.count() == len(chunks))

    # Release Chroma's file handles before the temp folder is removed (Windows)
    del collection, vectordb

print()
print(f"{len(clauses)} clauses, {len(chunks)} chunks, max {max(c['tokens'] for c in chunks)} tokens")

if failures:
    print(f"{len(failures)} check(s) failed")
    sys.exit(1)

print("All checks passed")

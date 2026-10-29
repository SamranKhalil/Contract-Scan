import re

# Tokens kept free below the model limit for [CLS], [SEP] and rounding
TOKEN_MARGIN = 16

# Abbreviations that end in a dot but do not end a sentence
ABBREVIATIONS = ["Ltd", "No", "Co", "Inc", "Corp", "Mr", "Mrs", "Ms", "Dr", "St", "etc", "e.g", "i.e", "vs"]

ABBREVIATION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in ABBREVIATIONS) + r")\.", re.IGNORECASE
)

PLACEHOLDER = "<DOT>"


# ---------------------------------------
# Guess clause type
# ---------------------------------------

# Word-boundary patterns, checked in order. Plain substring matching made
# "Relationship" and "Ownership" count as Intellectual Property ("ip").
CLAUSE_TYPE_PATTERNS = [
    (r"\bpreamble\b", "Preamble"),
    (r"\bdefinitions?\b|\binterpretation\b", "Definitions"),
    (r"\bliabilit", "Liability"),
    (r"\bindemn", "Indemnity"),
    (r"\bintellectual property\b|\bip\b", "Intellectual Property"),
    (r"\bconfidential", "Confidentiality"),
    (r"\bdata protection\b|\bpersonal data\b|\bgdpr\b|\bprivacy\b", "Data Protection"),
    (r"\bnon[- ]?compet|\brestrictive covenant", "Non-Compete"),
    (r"\bnon[- ]?solicit", "Non-Solicitation"),
    (r"\bterminat", "Termination"),
    (r"\bterm\b|\bduration\b|\brenewal\b", "Term"),
    (r"\bfees?\b|\bpayment\b|\bcharges\b|\bpricing\b|\binvoic", "Payment"),
    (r"\bwarrant", "Warranty"),
    (r"\bforce majeure\b", "Force Majeure"),
    (r"\bgoverning law\b|\bjurisdiction\b", "Governing Law"),
    (r"\bdisputes?\b|\barbitration\b", "Dispute Resolution"),
    (r"\bassignment\b|\bsubcontract", "Assignment"),
    (r"\baudit\b|\brecords\b", "Audit"),
    (r"\binsurance\b", "Insurance"),
    (r"\bnotices?\b", "Notices"),
    (r"\bservices?\b|\bscope\b|\bdeliverables?\b", "Services"),
    (r"\bgeneral\b|\bmiscellaneous\b|\bboilerplate\b", "General"),
]


def infer_clause_type(text):

    text = text.lower()

    for pattern, clause_type in CLAUSE_TYPE_PATTERNS:

        if re.search(pattern, text):

            return clause_type

    return "Other"


def classify_clause(heading, content):
    """
    Classifies by heading first. Headings like "Miscellaneous" give no real
    signal, so for those the clause text is used instead.
    """

    clause_type = infer_clause_type(heading)

    if clause_type in ("Other", "General"):

        from_content = infer_clause_type(content)

        if from_content != "Other":
            return from_content

    return clause_type


# ---------------------------------------
# Extract clause number
# ---------------------------------------


def extract_clause_number(heading):
    """
    "1. Definitions" -> "1", "Clause 7 Term" -> "7", "PREAMBLE" -> None.
    """

    match = re.match(r"^(?:clause\s+)?(\d+(?:\.\d+)*)", heading, re.IGNORECASE)

    if match:

        return match.group(1)

    return None


# ---------------------------------------
# Split very long clauses
# ---------------------------------------


def split_sentences(text):
    """
    Splits on sentence ends and semicolons, keeping "Ltd." and "No." intact.
    """

    protected = ABBREVIATION_PATTERN.sub(lambda m: m.group(1) + PLACEHOLDER, text)

    parts = re.split(r"(?<=[.!?;])\s+", protected)

    return [p.replace(PLACEHOLDER, ".").strip() for p in parts if p.strip()]


def split_words(text, fits):
    """
    Last resort for a single sentence that is still too long.
    """

    chunks = []
    current = ""

    for word in text.split():

        tentative = (current + " " + word).strip()

        if fits(tentative) or not current:
            current = tentative
        else:
            chunks.append(current)
            current = word

    if current:
        chunks.append(current)

    return chunks


def split_long_text(text, fits):
    """
    Splits text into pieces that each pass fits(), on sentence boundaries
    where possible and on word boundaries otherwise.
    """

    if not text or not text.strip():
        return []

    if fits(text):
        return [text.strip()]

    chunks = []
    current = ""

    for sentence in split_sentences(text):

        tentative = (current + " " + sentence).strip()

        if fits(tentative):
            current = tentative
            continue

        if current:
            chunks.append(current)
            current = ""

        if fits(sentence):
            current = sentence
        else:
            chunks.extend(split_words(sentence, fits))

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------
# Main chunker
# ---------------------------------------


def build_embed_text(heading, text):
    """
    The heading is prepended so continuation chunks keep their context.
    """

    return f"{heading}\n{text}"


def create_chunks(clauses, doc_id, embedder):
    """
    Converts parsed clause objects into flat chunk dicts ready for embedding.

    Each sub-clause becomes its own chunk (split further if too long), measured
    with the embedding model's own tokenizer so nothing gets truncated.
    Chunk IDs are prefixed with doc_id so contracts never overwrite each other.

    `text` stays raw for citation; `embed_text` is what gets embedded.
    """

    max_tokens = embedder.max_tokens - TOKEN_MARGIN

    chunks = []
    chunk_counter = 1

    for clause in clauses:

        heading = clause["heading"]
        clause_number = clause.get("clause_number") or extract_clause_number(heading)
        clause_type = classify_clause(heading, clause["content"])

        def fits(piece):
            return embedder.count_tokens(build_embed_text(heading, piece)) <= max_tokens

        for sub_clause in clause["sub_clauses"]:

            for piece in split_long_text(sub_clause["text"], fits):

                embed_text = build_embed_text(heading, piece)

                chunks.append(
                    {
                        "chunk_id": f"{doc_id}_{chunk_counter:04d}",
                        "doc_id": doc_id,
                        "clause_number": clause_number,
                        "sub_clause_number": sub_clause["number"],
                        "heading": heading,
                        "clause_type": clause_type,
                        "page": sub_clause["page"],
                        "tokens": embedder.count_tokens(embed_text),
                        "text": piece,
                        "embed_text": embed_text,
                    }
                )

                chunk_counter += 1

    return chunks

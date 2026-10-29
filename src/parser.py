import logging
import re
import statistics
from collections import defaultdict

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Share of the page height treated as header / footer margin
MARGIN_RATIO = 0.07

# Top-level clause headings: "1. Definitions", "1 Definitions", "Clause 1 ...",
# "SCHEDULE 1", "ANNEX A", "APPENDIX 2". Sub-clauses like "1.1" never match
# because the number must be followed by an optional dot and then whitespace.
HEADING_PATTERNS = [
    re.compile(r"^(\d+)\.?\s+[A-Z]"),
    re.compile(r"^Clause\s+(\d+)\b", re.IGNORECASE),
    # Whole label kept ("SCHEDULE 1") so it is not confused with clause 1
    re.compile(r"^((?:SCHEDULE|ANNEX|APPENDIX)\s+\w+)", re.IGNORECASE),
]

# Sub-clause at the start of a line: "3.1 The Client ..."
SUB_CLAUSE_PATTERN = re.compile(r"^(\d+)\.(\d+)\.?\s+")


# -------------------------------
# Text Cleaning
# -------------------------------


def clean_text(text):
    """
    Normalises whitespace in extracted PDF text.
    """

    text = text.replace(" ", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def join_lines(lines):
    """
    Joins PDF lines into one string.

    Lines are joined with a space so the last word of one line does not run
    into the first word of the next ("Supplier" + "shall"). A line ending in a
    hyphenated word rejoins its continuation without a space ("Agree-" + "ment").
    """

    text = ""

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if re.search(r"[A-Za-z]-$", text):
            text = text[:-1] + line
        elif text:
            text += " " + line
        else:
            text = line

    return clean_text(text)


# -------------------------------
# Extract lines with metadata
# -------------------------------


def extract_pdf_lines(pdf_bytes):
    """
    Takes PDF bytes and returns one record per text line.

    Output:
    [
        {
            page: 1,
            text: "...",
            font_size: 10.0,
            bold: True,          # True when most of the line is bold
            x0: 72.0,            # left edge of the line
            y0: 76.0,            # top of the line
            y1: 88.0,            # bottom of the line
            page_height: 841.9
        }
    ]
    """

    document = fitz.open(stream=pdf_bytes, filetype="pdf")

    lines = []

    for page_number, page in enumerate(document):

        page_height = page.rect.height
        page_dict = page.get_text("dict")
        page_lines = []

        for block in page_dict["blocks"]:

            # Ignore images
            if "lines" not in block:
                continue

            for line in block["lines"]:

                line_text = ""
                max_font_size = 0
                total_chars = 0
                bold_chars = 0

                for span in line["spans"]:

                    span_text = span["text"]
                    line_text += span_text
                    max_font_size = max(max_font_size, span["size"])

                    char_count = len(span_text.strip())
                    total_chars += char_count

                    # Flag 16 = bold in PyMuPDF; some fonts only say so in the name
                    if span["flags"] & 16 or "bold" in span["font"].lower():
                        bold_chars += char_count

                line_text = clean_text(line_text)

                if not line_text:
                    continue

                page_lines.append(
                    {
                        "page": page_number + 1,
                        "text": line_text,
                        "font_size": round(max_font_size, 1),
                        "x0": line["bbox"][0],
                        "y0": line["bbox"][1],
                        "y1": line["bbox"][3],
                        "page_height": page_height,
                        "_bold_chars": bold_chars,
                        "_total_chars": total_chars,
                    }
                )

        lines.extend(merge_rows(page_lines))

    document.close()

    for line in lines:
        total = line.pop("_total_chars")
        bold = line.pop("_bold_chars")
        line["bold"] = (bold / total) > 0.6 if total else False

    return lines


def merge_rows(page_lines):
    """
    Merges pieces of text that sit on the same visual row.

    Some PDFs store "1" and "DEFINITIONS" (or "1.1" and its text) as separate
    lines laid out with a tab stop, which hides headings and sub-clause numbers.
    """

    def center(line):
        return (line["y0"] + line["y1"]) / 2

    page_lines.sort(key=center)

    # Group pieces whose vertical centre falls inside the first piece's box
    groups = []

    for line in page_lines:

        if groups and groups[-1][0]["y0"] <= center(line) <= groups[-1][0]["y1"]:
            groups[-1].append(line)
        else:
            groups.append([line])

    rows = []

    for group in groups:

        group.sort(key=lambda l: l["x0"])

        row = dict(group[0])
        row["text"] = " ".join(l["text"] for l in group)
        row["font_size"] = max(l["font_size"] for l in group)
        row["y0"] = min(l["y0"] for l in group)
        row["y1"] = max(l["y1"] for l in group)
        row["_bold_chars"] = sum(l["_bold_chars"] for l in group)
        row["_total_chars"] = sum(l["_total_chars"] for l in group)

        rows.append(row)

    return rows


# -------------------------------
# Remove repeating headers/footers
# -------------------------------


def in_margin(line):

    margin = line["page_height"] * MARGIN_RATIO

    return line["y1"] <= margin or line["y0"] >= line["page_height"] - margin


def mask_digits(text):

    return re.sub(r"\d+", "#", text.lower())


def remove_headers_footers(lines):
    """
    Removes running headers and footers.

    Only lines in the top or bottom margin are candidates, so repeated body
    text like "Signature" or "Name:" is kept. Digits are masked before
    comparing, so "Page 1 of 6" and "Page 2 of 6" count as the same text.
    Bare page numbers in the margin are dropped too.
    """

    pages_per_text = defaultdict(set)

    for line in lines:
        if in_margin(line):
            pages_per_text[mask_digits(line["text"])].add(line["page"])

    repeated = {text for text, pages in pages_per_text.items() if len(pages) >= 2}

    cleaned = []

    for line in lines:

        if in_margin(line):

            masked = mask_digits(line["text"])
            is_page_number = re.fullmatch(r"(page\s*)?#(\s*(of|/)\s*#)?", masked)

            if masked in repeated or is_page_number:
                continue

        cleaned.append(line)

    return cleaned


# -------------------------------
# Detect headings
# -------------------------------


def heading_number(text):
    """
    Returns the clause number if the text looks like a top-level heading.
    """

    for pattern in HEADING_PATTERNS:

        match = pattern.match(text)

        if match:
            number = match.group(1)

            # "2026 The Parties..." or "250 Main Street" are not clause numbers
            if number.isdigit() and int(number) >= 100:
                return None

            return number

    return None


def detect_headings(lines, body_font, require_emphasis=True):
    """
    Marks top-level clause headings.

    A line is a heading when it matches a heading pattern and (unless
    require_emphasis is False) is bold or larger than the body font.
    """

    for line in lines:

        number = heading_number(line["text"])
        emphasised = line["font_size"] > body_font or line["bold"]

        line["is_heading"] = number is not None and (emphasised or not require_emphasis)
        line["clause_number"] = number if line["is_heading"] else None

    return lines


# -------------------------------
# Create Clause Objects
# -------------------------------


def new_clause(heading, clause_number, page):

    return {
        "heading": heading,
        "clause_number": clause_number,
        "page": page,
        "sub_clauses": [],
        # Lines of the sub-clause (or intro text) being built
        "_lines": [],
        "_number": None,
        "_page": page,
    }


def flush_sub_clause(clause):

    text = join_lines(clause["_lines"])

    if text:
        clause["sub_clauses"].append(
            {"number": clause["_number"], "text": text, "page": clause["_page"]}
        )

    clause["_lines"] = []


def finish_clause(clause, clauses):

    flush_sub_clause(clause)

    if not clause["sub_clauses"]:

        if clause["heading"] == "PREAMBLE":
            return

        # A heading with no body: keep it rather than silently dropping it
        logger.warning("Clause '%s' has no body text; keeping heading only", clause["heading"])
        clause["sub_clauses"].append(
            {"number": None, "text": clause["heading"], "page": clause["page"]}
        )

    for key in ("_lines", "_number", "_page"):
        del clause[key]

    clause["content"] = " ".join(s["text"] for s in clause["sub_clauses"])
    clauses.append(clause)


def build_clauses(lines):
    """
    Groups lines into clause objects.

    Text before the first heading becomes a PREAMBLE clause. Inside a clause,
    a line starting "N.M" where N is the clause's own number starts a new
    sub-clause, so a cross-reference to another clause does not split it.

    Output:
    [
        {
            heading: "6. LIMITATION OF LIABILITY",
            clause_number: "6",
            page: 3,
            content: "...",
            sub_clauses: [{number: "6.1", text: "...", page: 3}, ...]
        }
    ]
    """

    clauses = []

    current = new_clause("PREAMBLE", None, lines[0]["page"] if lines else 1)

    for line in lines:

        if line["is_heading"]:

            finish_clause(current, clauses)
            current = new_clause(line["text"], line["clause_number"], line["page"])
            continue

        match = SUB_CLAUSE_PATTERN.match(line["text"])

        if match and match.group(1) == current["clause_number"]:

            flush_sub_clause(current)
            current["_number"] = f"{match.group(1)}.{match.group(2)}"
            current["_page"] = line["page"]

        if not current["_lines"]:
            current["_page"] = line["page"]

        current["_lines"].append(line["text"])

    finish_clause(current, clauses)

    return clauses


# -------------------------------
# Main Parser Function
# -------------------------------


def parse_contract(pdf_bytes):

    lines = extract_pdf_lines(pdf_bytes)

    if not lines:
        logger.warning("No text found in PDF (it may be scanned); OCR is not supported yet")
        return []

    lines = remove_headers_footers(lines)

    # Median is robust to a few large titles skewing the baseline
    body_font = statistics.median(line["font_size"] for line in lines)

    lines = detect_headings(lines, body_font)

    if sum(line["is_heading"] for line in lines) < 3:
        logger.warning("Fewer than 3 headings found; retrying without font/bold check")
        lines = detect_headings(lines, body_font, require_emphasis=False)

    return build_clauses(lines)

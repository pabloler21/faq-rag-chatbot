from pathlib import Path

from src.chunking import chunk_document, load_document

TEXT = load_document(Path("data/faq_document.txt"))
CHUNKS = chunk_document(TEXT, chunk_size=220, overlap=35)


def test_generates_at_least_20_chunks():
    assert len(CHUNKS) >= 20


def test_every_chunk_within_token_bounds():
    bad = [(c.chunk_id, c.n_tokens) for c in CHUNKS if not 50 <= c.n_tokens <= 500]
    assert bad == [], f"chunks outside the 50-500 token range: {bad}"


def test_chunk_ids_are_sequential():
    assert [c.chunk_id for c in CHUNKS] == list(range(len(CHUNKS)))


def test_every_chunk_has_a_section():
    assert all(c.section for c in CHUNKS)


def test_full_document_coverage():
    """The union of chunk spans must cover every non-heading character."""
    covered = bytearray(len(TEXT))
    for c in CHUNKS:
        covered[c.char_start:c.char_end] = b"\x01" * (c.char_end - c.char_start)
    uncovered = [
        i for i, flag in enumerate(covered)
        if not flag and TEXT[i].strip() and not _is_heading_char(TEXT, i)
    ]
    assert uncovered == [], f"{len(uncovered)} characters lost between chunks"


def _is_heading_char(text: str, i: int) -> bool:
    line_start = text.rfind("\n", 0, i) + 1
    return text[line_start:line_start + 2] == "##" or text[line_start:line_start + 2] == "# "

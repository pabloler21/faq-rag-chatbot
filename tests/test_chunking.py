from pathlib import Path

from src.chunking import chunk_document, load_document

TEXT = load_document(Path("data/faq_document.txt"))
CHUNKS = chunk_document(TEXT, chunk_size=220, overlap=35)


def test_generates_at_least_20_chunks():
    assert len(CHUNKS) >= 20


def test_every_chunk_within_token_bounds():
    bad = [
        (chunk.chunk_id, chunk.n_tokens)
        for chunk in CHUNKS
        if not 50 <= chunk.n_tokens <= 500
    ]
    assert bad == [], f"chunks outside the 50-500 token range: {bad}"


def test_chunk_ids_are_sequential():
    assert [chunk.chunk_id for chunk in CHUNKS] == list(range(len(CHUNKS)))


def test_every_chunk_has_a_section():
    assert all(chunk.section for chunk in CHUNKS)


def test_full_document_coverage():
    """The union of chunk spans must cover every non-heading character."""
    covered = bytearray(len(TEXT))

    for chunk in CHUNKS:
        span_length = chunk.char_end - chunk.char_start
        covered[chunk.char_start:chunk.char_end] = b"\x01" * span_length

    uncovered = [
        i
        for i, flag in enumerate(covered)
        if not flag and TEXT[i].strip() and not _is_heading_char(TEXT, i)
    ]
    assert uncovered == [], f"{len(uncovered)} characters lost between chunks"


def _is_heading_char(text: str, i: int) -> bool:
    """True if position `i` sits on a '# ' or '## ' line, which no chunk stores."""
    line_start = text.rfind("\n", 0, i) + 1
    prefix = text[line_start:line_start + 2]
    return prefix in ("##", "# ")

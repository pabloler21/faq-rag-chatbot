"""Stage 1-2 of the indexing pipeline: load the document and split it into chunks."""
import re
import sys
from pathlib import Path

import tiktoken

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_FILE
from src.schemas import Chunk

_ENCODER = tiktoken.get_encoding("cl100k_base")
_HEADING = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*[ \t]*\n?")

MIN_CHUNK_TOKENS = 50


def load_document(path: Path) -> str:
    """Read the source document.

    Text mode already normalises CRLF to LF, which the char offsets depend on:
    every char_start/char_end in this module indexes into the string returned here.
    """
    return path.read_text(encoding="utf-8")


def count_tokens(text: str) -> int:
    """Approximate token count. cl100k_base is not the local model's tokenizer."""
    return len(_ENCODER.encode(text))


def _iter_sections(text: str) -> list[tuple[str, int, int]]:
    """Yield (section_name, body_start, body_end) for each '## ' heading."""
    matches = list(_HEADING.finditer(text))
    return [
        (m.group(1).strip(), m.end(), matches[i + 1].start() if i + 1 < len(matches) else len(text))
        for i, m in enumerate(matches)
    ]


def _iter_sentences(text: str, start: int, end: int) -> list[tuple[str, int, int]]:
    """Split a span into sentences, keeping absolute character offsets."""
    return [
        (m.group(), start + m.start(), start + m.end())
        for m in _SENTENCE.finditer(text[start:end])
        if m.group().strip()
    ]


def _tail(group: list, overlap: int) -> tuple[list, int]:
    """Take trailing sentences worth at most `overlap` tokens, for the next chunk."""
    kept: list = []
    total = 0
    for sentence in reversed(group):
        tokens = count_tokens(sentence[0])
        if total + tokens > overlap:
            break
        kept.insert(0, sentence)
        total += tokens
    return kept, total


def _merge_short_tail(groups: list[list]) -> list[list]:
    """Fold a final undersized group into its predecessor (rubric: min 50 tokens)."""
    if len(groups) > 1 and count_tokens("".join(s[0] for s in groups[-1])) < MIN_CHUNK_TOKENS:
        groups[-2] = groups[-2] + [s for s in groups[-1] if s not in groups[-2]]
        groups.pop()
    return groups


def _pack(sentences: list[tuple[str, int, int]], chunk_size: int, overlap: int) -> list[list]:
    """Greedily group sentences up to chunk_size, stepping back `overlap` tokens."""
    groups: list[list] = []
    current: list = []
    current_tokens = 0
    for sentence in sentences:
        tokens = count_tokens(sentence[0])
        if current and current_tokens + tokens > chunk_size:
            groups.append(current)
            current, current_tokens = _tail(current, overlap)
        current.append(sentence)
        current_tokens += tokens
    if current:
        groups.append(current)
    return _merge_short_tail(groups)


def chunk_document(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[Chunk]:
    """Split the document into chunks, preserving section context."""
    chunks: list[Chunk] = []
    for section, start, end in _iter_sections(text):
        for group in _pack(_iter_sentences(text, start, end), chunk_size, overlap):
            body = "".join(s[0] for s in group).strip()
            chunks.append(Chunk(
                chunk_id=len(chunks), text=body, section=section,
                n_tokens=count_tokens(body),
                char_start=group[0][1], char_end=group[-1][2],
            ))
    return chunks


def main() -> None:
    """Inspect chunking without spending a single model call."""
    chunks = chunk_document(load_document(DATA_FILE))
    sizes = [c.n_tokens for c in chunks]
    print(f"chunks : {len(chunks)}")
    print(f"tokens : min={min(sizes)} max={max(sizes)} avg={sum(sizes) // len(sizes)}")
    print(f"outside 50-500 : {[c.chunk_id for c in chunks if not 50 <= c.n_tokens <= 500]}")
    for c in chunks[:3]:
        print(f"\n[{c.chunk_id}] ({c.section}, {c.n_tokens}t) {c.text[:90]}...")


if __name__ == "__main__":
    sys.exit(main())

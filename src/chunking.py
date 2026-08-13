"""Stage 1-2 of the indexing pipeline: load the document and split it into chunks."""
import re
import sys
from pathlib import Path
from typing import NamedTuple

import tiktoken

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, DATA_FILE
from src.schemas import Chunk

_ENCODER = tiktoken.get_encoding("cl100k_base")
_HEADING = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]*[ \t]*\n?")

MIN_CHUNK_TOKENS = 50


class _Sentence(NamedTuple):
    """One sentence, with its absolute offsets in the source document.

    The offsets are what makes full-document coverage provable: the union of
    every chunk span must account for the whole file.
    """

    text: str
    start: int
    end: int


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
    """Return (section_name, body_start, body_end) for each '## ' heading.

    The body starts after the heading itself and ends where the next heading
    begins, or at the end of the document for the last section.
    """
    matches = list(_HEADING.finditer(text))
    sections: list[tuple[str, int, int]] = []

    for i, match in enumerate(matches):
        is_last = i + 1 == len(matches)
        name = match.group(1).strip()
        body_start = match.end()
        body_end = len(text) if is_last else matches[i + 1].start()
        sections.append((name, body_start, body_end))

    return sections


def _iter_sentences(text: str, start: int, end: int) -> list[_Sentence]:
    """Split one section into sentences, translating offsets to document coordinates."""
    sentences: list[_Sentence] = []

    for match in _SENTENCE.finditer(text[start:end]):
        if not match.group().strip():
            continue
        absolute_start = start + match.start()
        absolute_end = start + match.end()
        sentences.append(_Sentence(match.group(), absolute_start, absolute_end))

    return sentences


def _tail(group: list[_Sentence], overlap: int) -> tuple[list[_Sentence], int]:
    """Take trailing sentences worth at most `overlap` tokens, for the next chunk.

    This is the overlap mechanism: an answer split across two chunks stays whole
    in at least one of them.
    """
    kept: list[_Sentence] = []
    total = 0

    for sentence in reversed(group):
        tokens = count_tokens(sentence.text)
        if total + tokens > overlap:
            break
        kept.insert(0, sentence)
        total += tokens

    return kept, total


def _group_text(group: list[_Sentence]) -> str:
    """Join a group back into text. Sentences already carry their trailing space."""
    return "".join(sentence.text for sentence in group)


def _merge_short_tail(groups: list[list[_Sentence]]) -> list[list[_Sentence]]:
    """Fold a final undersized group into its predecessor (rubric: min 50 tokens).

    Only the last group can be undersized: every other one is closed by
    overflowing the budget.
    """
    if len(groups) < 2:
        return groups

    if count_tokens(_group_text(groups[-1])) >= MIN_CHUNK_TOKENS:
        return groups

    predecessor = groups[-2]
    # The tail opens with the predecessor's overlap, so those sentences are
    # already there -- merging them again would duplicate text inside one chunk.
    new_sentences = [s for s in groups[-1] if s not in predecessor]

    groups[-2] = predecessor + new_sentences
    groups.pop()
    return groups


def _pack(
    sentences: list[_Sentence],
    chunk_size: int,
    overlap: int,
) -> list[list[_Sentence]]:
    """Greedily group sentences up to chunk_size, stepping back `overlap` tokens."""
    groups: list[list[_Sentence]] = []
    current: list[_Sentence] = []
    current_tokens = 0

    for sentence in sentences:
        tokens = count_tokens(sentence.text)
        overflows = current and current_tokens + tokens > chunk_size

        if overflows:
            groups.append(current)
            current, current_tokens = _tail(current, overlap)

        # Runs on every iteration: the sentence that overflowed opens the next group.
        current.append(sentence)
        current_tokens += tokens

    if current:
        groups.append(current)

    return _merge_short_tail(groups)


def chunk_document(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Split the document into chunks, preserving section context."""
    chunks: list[Chunk] = []

    for section, body_start, body_end in _iter_sections(text):
        sentences = _iter_sentences(text, body_start, body_end)

        for group in _pack(sentences, chunk_size, overlap):
            body = _group_text(group).strip()
            chunks.append(
                Chunk(
                    # A global counter: chunk_id must equal the index in this list.
                    chunk_id=len(chunks),
                    text=body,
                    section=section,
                    n_tokens=count_tokens(body),
                    char_start=group[0].start,
                    char_end=group[-1].end,
                )
            )

    return chunks


def main() -> None:
    """Inspect chunking without spending a single model call."""
    chunks = chunk_document(load_document(DATA_FILE))
    sizes = [chunk.n_tokens for chunk in chunks]
    average = sum(sizes) // len(sizes)
    offenders = [chunk.chunk_id for chunk in chunks if not 50 <= chunk.n_tokens <= 500]

    print(f"chunks : {len(chunks)}")
    print(f"tokens : min={min(sizes)} max={max(sizes)} avg={average}")
    print(f"outside 50-500 : {offenders}")

    for chunk in chunks[:3]:
        preview = chunk.text[:90]
        print(f"\n[{chunk.chunk_id}] ({chunk.section}, {chunk.n_tokens}t) {preview}...")


if __name__ == "__main__":
    sys.exit(main())

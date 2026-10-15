"""Ingestion: load Markdown -> clean -> chunk -> embed -> store in ChromaDB."""
import hashlib
import re
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    Language,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "tech_docs"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 800      # characters
CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 50  # drop tiny leftovers

HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


# ---------- 1. Clean ----------
def clean_markdown(raw: str) -> tuple[str, str]:
    """Return (cleaned_text, source_url)."""
    lines = raw.splitlines()
    url = ""
    if lines and lines[0].startswith("Source:"):
        url = lines[0].removeprefix("Source:").strip()
        lines = lines[1:]
    text = "\n".join(lines)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)                  # image lines
    text = re.sub(r"^Python 3\.\d+\+\s*$", "", text, flags=re.MULTILINE)  # tab labels
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip(), url


# ---------- 2. Chunk ----------
def chunk_markdown(text: str, doc_name: str, url: str) -> list[Document]:
    # First split by headings so each chunk stays inside one topic
    header_splitter = MarkdownHeaderTextSplitter(HEADERS, strip_headers=False)
    sections = header_splitter.split_text(text)

    # Then split long sections by size, respecting code fences
    size_splitter = RecursiveCharacterTextSplitter.from_language(
        Language.MARKDOWN, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    pieces = size_splitter.split_documents(sections)

    chunks, seen = [], set()
    for piece in pieces:
        body = piece.page_content.strip()
        if len(body) < MIN_CHUNK_CHARS:
            continue
        digest = hashlib.md5(body.encode()).hexdigest()
        if digest in seen:          # the docs repeat the same code example many times
            continue
        seen.add(digest)

        section = " > ".join(
            piece.metadata[k] for k in ("h1", "h2", "h3") if k in piece.metadata
        )
        # Put the heading path inside the text so it helps retrieval too
        content = f"[{section}]\n{body}" if section else body
        chunks.append(
            Document(
                page_content=content,
                metadata={
                    "doc_name": doc_name,
                    "source_url": url,
                    "section": section,
                    "chunk_index": len(chunks),
                },
            )
        )
    return chunks


# ---------- 3. Embed + store ----------
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def ingest_text(raw: str, doc_name: str, url: str | None = None) -> int:
    """Chunk and index one document. Re-ingesting the same name replaces it."""
    text, found_url = clean_markdown(raw)
    chunks = chunk_markdown(text, doc_name, url or found_url)

    vs = get_vectorstore()
    old_ids = vs.get(where={"doc_name": doc_name})["ids"]
    if old_ids:
        vs.delete(ids=old_ids)

    ids = [f"{doc_name}-{c.metadata['chunk_index']}" for c in chunks]
    vs.add_documents(chunks, ids=ids)
    return len(chunks)


def ingest_file(path: Path) -> int:
    return ingest_text(path.read_text(encoding="utf-8"), path.stem)


def ingest_directory(data_dir: Path = DATA_DIR) -> dict[str, int]:
    return {p.name: ingest_file(p) for p in sorted(data_dir.glob("*.md"))}


def list_documents() -> list[dict]:
    """Used later by GET /documents."""
    metas = get_vectorstore().get(include=["metadatas"])["metadatas"]
    docs: dict[str, dict] = {}
    for m in metas:
        d = docs.setdefault(
            m["doc_name"],
            {"doc_name": m["doc_name"], "source_url": m["source_url"], "chunks": 0},
        )
        d["chunks"] += 1
    return sorted(docs.values(), key=lambda d: d["doc_name"])


if __name__ == "__main__":
    for name, n in ingest_directory().items():
        print(f"{name}: {n} chunks")
    print("\nIndexed documents:")
    for d in list_documents():
        print(" ", d)
"""FastAPI app exposing the RAG workflow."""
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from groq import APIStatusError, RateLimitError
from markdownify import markdownify as md

from app.graph import run_query
from app.ingestion import ROOT, ingest_text, list_documents
from app.schemas import (
    DocumentInfo,
    FeedbackRequest,
    FeedbackResponse,
    IngestResponse,
    IngestUrlRequest,
    QueryRequest,
    QueryResponse,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("rag.api")

app = FastAPI(
    title="RAG Technical Documentation Assistant",
    description="Self-corrective RAG built with LangGraph.",
    version="1.0.0",
)

FEEDBACK_FILE = ROOT / "feedback.jsonl"
MAX_UPLOAD_BYTES = 2_000_000
ALLOWED_SUFFIXES = {".md", ".txt", ".html", ".htm"}


def _safe_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    return name[:60] or "document"


def _html_to_markdown(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    node = soup.find("article") or soup.find("main") or soup.body or soup
    for tag in node.select("script, style, nav, button, a.headerlink"):
        tag.decompose()
    return md(str(node), heading_style="ATX")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    question = req.question.strip()
    if len(question) < 3:
        raise HTTPException(status_code=422, detail="Question is too short.")
    try:
        result = run_query(question)
    except RateLimitError:
        raise HTTPException(status_code=503, detail="LLM rate limit reached. Try again shortly.")
    except APIStatusError as e:
        log.exception("LLM provider error")
        raise HTTPException(status_code=502, detail=f"LLM provider error ({e.status_code}).")
    except Exception:
        log.exception("Query failed")
        raise HTTPException(status_code=500, detail="Internal error while answering.")
    return QueryResponse(query_id=uuid.uuid4().hex[:12], **result)


@app.post("/ingest", response_model=IngestResponse, status_code=201)
async def ingest_file_upload(file: UploadFile = File(...)):
    """Upload a .md, .txt or .html file."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported file type. Use: {sorted(ALLOWED_SUFFIXES)}")
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 2 MB).")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 text.")
    if suffix in {".html", ".htm"}:
        text = _html_to_markdown(text)
    name = _safe_name(Path(file.filename).stem)
    n = ingest_text(text, name, url=f"upload:{file.filename}")
    if n == 0:
        raise HTTPException(status_code=422, detail="No usable text found in the file.")
    return IngestResponse(doc_name=name, chunks=n)


@app.post("/ingest/url", response_model=IngestResponse, status_code=201)
def ingest_from_url(req: IngestUrlRequest):
    """Fetch a web page and index it."""
    url = str(req.url)
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "rag-assistant"})
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch URL: {e}")
    if len(resp.content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Page too large (max 2 MB).")
    text = _html_to_markdown(resp.text)
    name = _safe_name(req.name or url.rstrip("/").split("/")[-1])
    n = ingest_text(text, name, url=url)
    if n == 0:
        raise HTTPException(status_code=422, detail="No usable text found at that URL.")
    return IngestResponse(doc_name=name, chunks=n)


@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    return list_documents()


@app.post("/feedback", response_model=FeedbackResponse, status_code=201)
def feedback(req: FeedbackRequest):
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        **req.model_dump(),
    }
    with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return FeedbackResponse()
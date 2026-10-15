"""LangGraph workflow: query analysis -> retrieve -> grade -> (generate | rewrite | fallback)."""
import logging
import os
import re
from functools import lru_cache

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

from app.ingestion import get_vectorstore
from app.prompts import (
    GENERATE_PROMPT,
    GRADE_PROMPT,
    QUERY_ANALYSIS_PROMPT,
    REWRITE_PROMPT,
)
from app.state import GraphState

load_dotenv(override=True)
log = logging.getLogger("rag.graph")

TOP_K = 5
MAX_RETRIES = 2   # rewrites allowed, so at most 3 retrievals in total
QUERY_TYPES = {"conceptual", "how-to", "troubleshooting", "api-reference"}
FALLBACK_ANSWER = (
    "I couldn't find relevant information about that in the indexed documentation."
)


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        max_retries=3,
    )


def _chain(template: str):
    return ChatPromptTemplate.from_template(template) | get_llm() | StrOutputParser()


# ---------------- Nodes ----------------
def query_analysis(state: GraphState) -> dict:
    raw = _chain(QUERY_ANALYSIS_PROMPT).invoke({"question": state["question"]})
    type_match = re.search(r"TYPE:\s*(.+)", raw, re.IGNORECASE)
    query_match = re.search(r"QUERY:\s*(.+)", raw, re.IGNORECASE)

    query_type = type_match.group(1).strip().lower() if type_match else "unknown"
    if query_type not in QUERY_TYPES:
        query_type = "unknown"
    # If the LLM reply is malformed, fall back to the raw question
    search_query = query_match.group(1).strip() if query_match else state["question"]

    log.info("[query_analysis] type=%s | query=%s", query_type, search_query)
    return {
        "query_type": query_type,
        "search_query": search_query,
        "query_history": [search_query],
    }


def retrieve(state: GraphState) -> dict:
    results = get_vectorstore().similarity_search_with_score(
        state["search_query"], k=TOP_K
    )
    docs = []
    for doc, score in results:
        doc.metadata["score"] = round(float(score), 3)
        docs.append(doc)
    log.info("[retrieve] attempt %d | %d chunks", state["retry_count"] + 1, len(docs))
    return {"documents": docs}


def grade_documents(state: GraphState) -> dict:
    docs = state["documents"]
    if not docs:
        return {"relevant_documents": []}

    # Grade against the ORIGINAL question, one LLM call per chunk
    inputs = [{"question": state["question"], "document": d.page_content} for d in docs]
    verdicts = _chain(GRADE_PROMPT).batch(inputs, config={"max_concurrency": 4})

    relevant = []
    for doc, verdict in zip(docs, verdicts):
        keep = verdict.strip().lower().startswith("yes")
        log.info(
            "[grade] %s | %s | %s",
            "KEEP" if keep else "DROP",
            doc.metadata["doc_name"],
            doc.metadata["section"],
        )
        if keep:
            relevant.append(doc)
    log.info("[grade] %d of %d relevant", len(relevant), len(docs))
    return {"relevant_documents": relevant}


def rewrite_query(state: GraphState) -> dict:
    previous = "\n".join(f"- {q}" for q in state["query_history"])
    raw = _chain(REWRITE_PROMPT).invoke(
        {"question": state["question"], "previous_queries": previous}
    )
    lines = [ln.strip().strip('"') for ln in raw.splitlines() if ln.strip()]
    new_query = lines[0] if lines else state["question"]

    log.info("[rewrite] retry %d | new query: %s", state["retry_count"] + 1, new_query)
    return {
        "search_query": new_query,
        "query_history": [new_query],
        "retry_count": state["retry_count"] + 1,
    }


def generate(state: GraphState) -> dict:
    docs = state["relevant_documents"]
    context = "\n\n".join(
        f"Source {i} ({d.metadata['doc_name']} > {d.metadata['section']}):\n{d.page_content}"
        for i, d in enumerate(docs, start=1)
    )
    answer = _chain(GENERATE_PROMPT).invoke(
        {"question": state["question"], "context": context}
    )

    # Return only the sources the answer actually cites (all of them if none are cited)
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    used = [i for i in range(1, len(docs) + 1) if i in cited] or list(range(1, len(docs) + 1))
    sources = [
        {
            "id": i,
            "doc_name": docs[i - 1].metadata["doc_name"],
            "section": docs[i - 1].metadata["section"],
            "source_url": docs[i - 1].metadata["source_url"],
            "score": docs[i - 1].metadata["score"],
        }
        for i in used
    ]
    log.info("[generate] answered using %d sources", len(sources))
    return {"generation": answer.strip(), "sources": sources, "status": "answered"}


def fallback(state: GraphState) -> dict:
    log.info("[fallback] no relevant documents after %d retries", state["retry_count"])
    return {"generation": FALLBACK_ANSWER, "sources": [], "status": "no_answer"}


# ---------------- Conditional edge ----------------
def route_after_grading(state: GraphState) -> str:
    if state["relevant_documents"]:
        return "generate"
    if state["retry_count"] < MAX_RETRIES:
        return "rewrite_query"
    return "fallback"


# ---------------- Build ----------------
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("query_analysis", query_analysis)
    g.add_node("retrieve", retrieve)
    g.add_node("grade_documents", grade_documents)
    g.add_node("rewrite_query", rewrite_query)
    g.add_node("generate", generate)
    g.add_node("fallback", fallback)

    g.add_edge(START, "query_analysis")
    g.add_edge("query_analysis", "retrieve")
    g.add_edge("retrieve", "grade_documents")
    g.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query", "fallback": "fallback"},
    )
    g.add_edge("rewrite_query", "retrieve")   # the retry loop
    g.add_edge("generate", END)
    g.add_edge("fallback", END)
    return g.compile()


graph = build_graph()


def run_query(question: str) -> dict:
    """Run the workflow once. This is what the FastAPI endpoint will call."""
    result = graph.invoke(
        {
            "question": question,
            "search_query": question,
            "query_type": "",
            "query_history": [],
            "documents": [],
            "relevant_documents": [],
            "retry_count": 0,
            "generation": "",
            "sources": [],
            "status": "",
        }
    )
    return {
        "answer": result["generation"],
        "sources": result["sources"],
        "status": result["status"],
        "query_type": result["query_type"],
        "retries": result["retry_count"],
        "queries_tried": result["query_history"],
    }
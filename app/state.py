"""The state that flows through the LangGraph workflow."""
import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    question: str                    # the user's original question (never modified)
    search_query: str                # the current query used for retrieval
    query_type: str                  # conceptual | how-to | troubleshooting | api-reference
    query_history: Annotated[list[str], operator.add]   # every query tried (reducer: append)
    documents: list[Document]        # chunks returned by retrieval
    relevant_documents: list[Document]   # chunks that passed grading
    retry_count: int                 # how many query rewrites so far
    generation: str                  # the answer text
    generation_attempts: int         # how many times generate has run
    grounding: str                   # supported | unsupported | unverified | not_applicable
    grounding_note: str              # which claims were unsupported (if any)
    sources: list[dict]              # citations returned to the user
    status: str                      # "answered" or "no_answer"
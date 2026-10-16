import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)

from app.graph import graph, run_query

QUESTIONS = [
    "How do I accept JSON data in a POST request?",
    "How do I return a 404 error?",
    "How do I limit a query parameter to a maximum length of 50 characters?",  # likely beyond the docs
    "Best pasta recipe?",
]

for q in QUESTIONS:
    print("\n" + "=" * 70)
    print("Q:", q)
    result = run_query(q)
    print("\nSTATUS:", result["status"], "| retries:", result["retries"],
          "| generations:", result["generation_attempts"])
    print("GROUNDING:", result["grounding"], "|", result["grounding_note"])
    print("ANSWER:", result["answer"])
    print("SOURCES:", json.dumps(result["sources"], indent=2))

print("\n" + graph.get_graph().draw_mermaid())
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ingestion import get_vectorstore

QUESTIONS = [
    "How do I accept JSON data in a POST request?",
    "How do I return a 404 error?",
    "What is dependency injection?",
    "How do I make a query parameter optional?",
    "Best pasta recipe?",   # off-topic on purpose
]

vs = get_vectorstore()
for q in QUESTIONS:
    print(f"\nQ: {q}")
    for doc, score in vs.similarity_search_with_score(q, k=3):
        print(f"  {score:.3f} | {doc.metadata['doc_name']} | {doc.metadata['section']}")
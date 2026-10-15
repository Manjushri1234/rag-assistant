from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

MAX_RETRIES = 2


# 1. STATE: what data flows between nodes
class State(TypedDict):
    question: str
    documents: List[str]
    relevant_documents: List[str]
    retry_count: int
    answer: str


# 2. NODES: each returns only what it changes
def retrieve(state: State):
    q = state["question"].lower()
    print(f"[retrieve] attempt {state['retry_count']} | question: {state['question']}")
    # Fake search: good docs only after a retry, and never for pasta questions
    if "pasta" not in q and state["retry_count"] >= 1:
        docs = ["FastAPI uses Pydantic for validation",
                "Pydantic models define request bodies"]
    else:
        docs = ["Pasta cooking tips", "Football scores"]
    return {"documents": docs}


def grade(state: State):
    # Real version: an LLM says relevant/irrelevant per chunk
    relevant = [d for d in state["documents"] if "pydantic" in d.lower()]
    print(f"[grade] {len(relevant)} of {len(state['documents'])} relevant")
    return {"relevant_documents": relevant}


def rewrite(state: State):
    print("[rewrite] improving the query")
    return {
        "question": state["question"] + " (rewritten)",
        "retry_count": state["retry_count"] + 1,
    }


def generate(state: State):
    print("[generate] answering from relevant docs")
    return {"answer": "Answer based on: " + "; ".join(state["relevant_documents"])}


def fallback(state: State):
    print("[fallback] giving up")
    return {"answer": "I don't know based on the documents."}


# 3. CONDITIONAL EDGE: returns the NAME of the next node
def route_after_grading(state: State):
    if state["relevant_documents"]:
        return "generate"
    if state["retry_count"] < MAX_RETRIES:
        return "rewrite"
    return "fallback"


# 4. BUILD THE GRAPH
builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade)
builder.add_node("rewrite", rewrite)
builder.add_node("generate", generate)
builder.add_node("fallback", fallback)

builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges(
    "grade",
    route_after_grading,
    {"generate": "generate", "rewrite": "rewrite", "fallback": "fallback"},
)
builder.add_edge("rewrite", "retrieve")   # this arrow creates the retry loop
builder.add_edge("generate", END)
builder.add_edge("fallback", END)

graph = builder.compile()


def run(question: str):
    print("\n" + "=" * 50)
    result = graph.invoke({
        "question": question,
        "documents": [],
        "relevant_documents": [],
        "retry_count": 0,
        "answer": "",
    })
    print("FINAL ANSWER:", result["answer"])
    print("Retries used:", result["retry_count"])


if __name__ == "__main__":
    run("How does FastAPI validate data?")   # fails once, retries, succeeds
    run("Best pasta recipe?")                # never succeeds, hits retry limit
    print("\n" + graph.get_graph().draw_mermaid())
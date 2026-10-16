"""Prompt templates. Kept separate so they are easy to read and tune."""

QUERY_ANALYSIS_PROMPT = """You prepare questions for searching FastAPI technical documentation.

1. Classify the question as exactly one of: conceptual, how-to, troubleshooting, api-reference.
2. Rewrite it as a short search query: clarify vague wording and add relevant technical terms and synonyms. Maximum 25 words.

Reply in exactly this format and nothing else:
TYPE: <type>
QUERY: <query>

Question: {question}"""

GRADE_BATCH_PROMPT = """You are grading documentation excerpts for relevance to a question.
For each excerpt, decide whether it contains information that helps answer the question, even partially.
Mark false if it is unrelated, or only shares keywords without being useful.

Reply with ONLY a JSON array of {n} booleans, in the same order as the excerpts. Example: [true, false, true]

Question: {question}

{excerpts}"""

REWRITE_PROMPT = """A search over FastAPI documentation did not return useful results.
Write ONE new search query for the original question that approaches it from a different angle:
use different terminology, more specific technical terms, or a broader concept.
Do not repeat any previous query.

Original question: {question}

Previous queries that failed:
{previous_queries}

Reply with only the new query, on a single line."""

GENERATE_PROMPT = """You are a technical documentation assistant. Answer the question using ONLY the sources below.

Rules:
- If the sources do not contain the answer, say so. Do not use outside knowledge.
- Cite sources ONLY by number, like [1] or [2][3]. Never write section names in brackets.
- Code examples must be copied or adapted from the sources. Do not invent fields, functions or parameters that the sources don't contain.
- Be clear and concise.
{feedback}
Question: {question}

Sources:
{context}"""

GROUNDING_PROMPT = """You are a strict fact checker. Decide whether an ANSWER is fully supported by the SOURCES.

Check the factual claims, and any functions, parameters, fields or behaviours used in code.
Ignore: wording, formatting, citation markers like [1], and obvious placeholder names in code
(for example a made-up helper like get_item_from_db that is clearly a stand-in).
Flag anything that states or relies on a fact that is NOT in the sources.

Reply with ONLY a JSON object, no other text:
{{"supported": true or false, "unsupported_claims": ["short description", ...]}}

SOURCES:
{sources}

ANSWER:
{answer}"""
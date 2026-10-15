"""Prompt templates. Kept separate so they are easy to read and tune."""

QUERY_ANALYSIS_PROMPT = """You prepare questions for searching FastAPI technical documentation.

1. Classify the question as exactly one of: conceptual, how-to, troubleshooting, api-reference.
2. Rewrite it as a short search query: clarify vague wording and add relevant technical terms and synonyms. Maximum 25 words.

Reply in exactly this format and nothing else:
TYPE: <type>
QUERY: <query>

Question: {question}"""

GRADE_PROMPT = """You are grading whether a documentation excerpt helps answer a question.

Answer "yes" if the excerpt contains information that helps answer the question, even partially.
Answer "no" if it is unrelated, or only shares a few keywords without being useful.

Reply with one word: yes or no.

Question: {question}

Excerpt:
{document}"""

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
- Cite the sources you used with their numbers in square brackets, like [1] or [2][3].
- Be clear and concise. Include a short code example when the sources contain one that helps.

Question: {question}

Sources:
{context}"""
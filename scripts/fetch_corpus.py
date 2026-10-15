"""Download a small FastAPI docs corpus into data/ as Markdown files."""
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PAGES = {
    "first-steps": "https://fastapi.tiangolo.com/tutorial/first-steps/",
    "query-params": "https://fastapi.tiangolo.com/tutorial/query-params/",
    "request-body": "https://fastapi.tiangolo.com/tutorial/body/",
    "handling-errors": "https://fastapi.tiangolo.com/tutorial/handling-errors/",
    "dependencies": "https://fastapi.tiangolo.com/tutorial/dependencies/",
}


def fetch_page(name: str, url: str) -> None:
    resp = requests.get(url, timeout=30, headers={"User-Agent": "rag-assistant-corpus-fetcher"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    article = soup.find("article") or soup.find("main")
    if article is None:
        raise RuntimeError(f"Could not find main content for {url}")

    # Remove noise: anchor icons, buttons, navigation bits
    for tag in article.select("a.headerlink, button, nav, script, style"):
        tag.decompose()

    text = md(str(article), heading_style="ATX", code_language="python")
    text = "\n".join(line.rstrip() for line in text.splitlines())
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    out = DATA_DIR / f"{name}.md"
    # Keep the source URL at the top: useful for citations later
    out.write_text(f"Source: {url}\n\n{text.strip()}\n", encoding="utf-8")
    print(f"Saved {out.name}: {len(text):,} characters")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for page_name, page_url in PAGES.items():
        fetch_page(page_name, page_url)
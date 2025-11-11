"""
ChatNShop Orchestrator
- Connects Qdrant (semantic search)
- Embeds queries using FastEmbed
- Optionally uses Ollama (Llama3) for natural responses
"""

import os
import json
import time
import httpx
import inspect
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Callable, TypeVar, Coroutine, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding
from dotenv import load_dotenv

# ============================
# Load .env file
# ============================
load_dotenv()

# ============================
# Configuration
# ============================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "semantic_collection")

USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# ============================
# Initialize App & Clients
# ============================
app = FastAPI(title="ChatNShop Orchestrator", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
embedder = TextEmbedding(model_name=EMBED_MODEL)

print("\n🚀 Starting ChatNShop Orchestrator...")
print(f"🔹 Qdrant Host: {QDRANT_HOST}:{QDRANT_PORT}")
print(f"🔹 Collection: {QDRANT_COLLECTION}")
print(f"🔹 Ollama Enabled: {USE_OLLAMA}")
print(f"🔹 Ollama Model: {OLLAMA_MODEL}")
print(f"🔹 Embed Model: {EMBED_MODEL}\n")

# ============================
# Request Schema
# ============================
class QueryIn(BaseModel):
    query: str
    top_k: int = 5

# ============================
# Logging Decorator
# ============================

from typing import Callable, TypeVar, Any
import inspect
import time

F = TypeVar("F", bound=Callable[..., Any])

def log_llm_call(name: str) -> Callable[[F], F]:
    """Logs both sync and async function calls safely without redeclaration warnings."""
    def decorator(func: F) -> F:
        if inspect.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                print(f"🟢 [AsyncLog] {name} called with args={args}, kwargs={kwargs}")
                result = await func(*args, **kwargs)
                print(f"✅ [AsyncLog] {name} completed in {time.time() - start:.2f}s")
                return result
            return async_wrapper  # type: ignore
        else:
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                print(f"🟢 [Log] {name} called with args={args}, kwargs={kwargs}")
                result = func(*args, **kwargs)
                print(f"✅ [Log] {name} completed in {time.time() - start:.2f}s")
                return result
            return sync_wrapper  # type: ignore
    return decorator

# ============================
# Intent Classifier
# ============================
def classify_intent(q: str) -> str:
    ql = q.lower()
    if any(k in ql for k in ["price", "cost", "discount", "offer"]):
        return "pricing"
    if any(k in ql for k in ["stock", "available", "availability"]):
        return "availability"
    if any(k in ql for k in ["feature", "spec", "details", "material", "size"]):
        return "product_info"
    return "general"

# ============================
# Embedding and Search
# ============================
def embed_text(text: str):
    return list(embedder.embed([text]))[0]

@log_llm_call("semantic_search")
def semantic_search(query: str, top_k: int) -> List[Dict[str, Any]]:
    vector = embed_text(query)
    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=vector,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    results = []
    for h in hits:
        p = h.payload or {}
        results.append({
            "score": float(h.score),
            "title": p.get("title") or p.get("name"),
            "description": p.get("description"),
            "price": p.get("price"),
            "currency": p.get("currency") or "INR",
            "url": p.get("url") or "#",
            "image": p.get("image"),
            "vendor": p.get("vendor"),
        })
    return results

# ============================
# HTML Response Formatter
# ============================
@log_llm_call("formatter")
def format_html(query: str, intent: str, results: List[Dict[str, Any]]) -> str:
    def card(r):
        img = f'<img src="{r["image"]}" alt="" style="max-width:120px;border-radius:10px;"/>' if r.get("image") else ""
        price = f'{r.get("currency","")} {r.get("price","")}'
        return f"""
        <div style="display:flex;gap:10px;margin-bottom:12px;border:1px solid #eee;padding:10px;border-radius:10px;">
            {img}
            <div>
                <h3 style="margin:0;">{r.get("title","(untitled)")}</h3>
                <p style="opacity:.85;margin:4px 0;">{(r.get("description") or "")[:200]}</p>
                <p><b>{price}</b> - <i>{r.get("vendor","")}</i></p>
                <a href="{r.get('url','#')}" target="_blank">View Product</a>
            </div>
        </div>"""
    cards = "".join(card(r) for r in results)
    return f"""
    <section style="font-family:Arial, sans-serif;">
        <h2>Results for “{query}”</h2>
        <p style="opacity:.7;">Intent: <b>{intent}</b> • Top {len(results)} results</p>
        {cards or "<p>No results found.</p>"}
    </section>
    """

# ============================
# Ollama (LLM) Connector
# ============================
@log_llm_call("ollama_html")
async def ollama_html(query: str, intent: str, results: List[Dict[str, Any]]) -> str:
    system_prompt = (
        "You are a helpful e-commerce assistant. Return ONLY HTML. "
        "Make a neat product list with title, description, price, and link."
    )
    user_prompt = f"""
User query: {query}
Intent: {intent}
Products JSON:
{json.dumps(results, indent=2, ensure_ascii=False)}
"""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
        )
        data = resp.json()
        content = (
            (data.get("message") or {}).get("content")
            or ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
            or ""
        )
        return content.strip() or format_html(query, intent, results)

# ============================
# Routes
# ============================
@app.get("/health", response_class=HTMLResponse)
def health():
    return "<b>OK</b>"

@app.get("/", response_class=JSONResponse)
def root():
    return {"status": "up", "service": "ChatNShop Orchestrator"}

@app.post("/api/query", response_class=HTMLResponse)
async def handle_query(data: QueryIn):
    query = (data.query or "").strip()
    if not query:
        return HTMLResponse("<p>Please enter a query.</p>", status_code=400)

    intent = classify_intent(query)
    try:
        results = semantic_search(query, data.top_k)
    except Exception as e:
        return HTMLResponse(
            f"<p>Search failed. Check Qdrant is running and collection '{QDRANT_COLLECTION}' exists.<br>Error: {e}</p>",
            status_code=500,
        )

    if USE_OLLAMA:
        try:
            html = await ollama_html(query, intent, results)
        except Exception as e:
            html = f"<p>Ollama error: {e}</p>" + format_html(query, intent, results)
    else:
        html = format_html(query, intent, results)

    return HTMLResponse(content=html, status_code=200)

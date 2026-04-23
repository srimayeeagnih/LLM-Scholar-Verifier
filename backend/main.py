"""
FastAPI backend for the Academic Claim Verifier.
Wraps MCP_test_extension.py pipeline with SSE streaming.
"""
import sys
import os
import json
import queue
import threading
import io

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Allow importing MCP_test_extension from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from MCP_test_extension import search_by_claims, compute_claim_verdict
from faiss_cache import FAISSQueryCache

app = FastAPI(title="Academic Claim Verifier")

# Initialised once at startup — embedding model loads here, not per-request
_cache = FAISSQueryCache()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class VerifyRequest(BaseModel):
    query: str


class QueueWriter(io.TextIOBase):
    """Redirects stdout to a queue so we can stream it as SSE."""
    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, text: str) -> int:
        if text.strip():
            self.q.put(("progress", text.rstrip()))
        return len(text)

    def flush(self):
        pass


def _build_response(claim_results: list) -> list:
    output = []
    for claim_data in claim_results:
        verdict = compute_claim_verdict(claim_data)
        ct = verdict["claim_totals"]
        papers = claim_data["results"].get("combined", [])[:10]

        paper_list = []
        for p in papers:
            paper_list.append({
                "title": p.get("title", ""),
                "authors": p.get("authors", [])[:5],
                "abstract": (p.get("abstract", "") or "")[:500],
                "source": p.get("source", ""),
                "pdf_url": p.get("pdf_url", ""),
                "best_similarity": round(p.get("best_similarity", 0), 3),
                "sentence_comparisons": [
                    {
                        "sentence": c["sentence"][:200],
                        "verdict": c["verdict"],
                        "sim_original": round(c["sim_original"], 3),
                        "sim_counterfactual": round(c["sim_counterfactual"], 3),
                    }
                    for c in p.get("sentence_comparisons", [])[:3]
                ],
                "negation_scores": p.get("negation_scores", {}),
            })

        output.append({
            "claim_text": claim_data["claim_text"],
            "keywords": claim_data["keywords"],
            "counterfactual": claim_data["results"].get("counterfactual", ""),
            "verdict": {
                "agree": ct["agree"],
                "disagree": ct["disagree"],
                "neutral": ct["neutral"],
                "total": ct["total"],
                "agreement_rate": round(ct["agreement_rate"], 3),
                "disagreement_rate": round(ct["disagreement_rate"], 3),
                "validity_score": round(ct["validity_score"], 3),
            },
            "papers": paper_list,
        })

    return output


@app.post("/api/verify")
async def verify(request: VerifyRequest):
    q: queue.Queue = queue.Queue()

    def run_pipeline():
        old_stdout = sys.stdout
        sys.stdout = QueueWriter(q)
        try:
            # ── Cache lookup ──────────────────────────────────────────────
            hit = _cache.search(request.query)
            if hit is not None:
                cached_result, score = hit
                print(f"[cache] HIT  (similarity={score:.3f}) — skipping pipeline")
                q.put(("result", json.dumps(cached_result)))
                return

            # ── Full pipeline ─────────────────────────────────────────────
            print(f"[cache] MISS — running full pipeline  (cache size={_cache.size()})")
            results = search_by_claims(request.query)
            response_data = _build_response(results)
            _cache.store(request.query, response_data)
            q.put(("result", json.dumps(response_data)))
        except Exception as e:
            q.put(("error", str(e)))
        finally:
            sys.stdout = old_stdout
            q.put(("done", ""))

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    def event_stream():
        while True:
            msg_type, msg_data = q.get()
            yield f"data: {json.dumps({'type': msg_type, 'data': msg_data})}\n\n"
            if msg_type in ("done", "error"):
                break

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok"}

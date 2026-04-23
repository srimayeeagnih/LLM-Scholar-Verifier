"""
LLM Academic Claim Verifier — Web UI

Flask app that:
  1. Accepts an LLM claim via a web UI
  2. Runs the paper search pipeline (arXiv + Semantic Scholar)
  3. Streams real-time progress to the browser via SSE
  4. Returns verdict scores and structured paper data
"""

import io
import sys
import json
import threading
import queue
from flask import Flask, request, jsonify, send_file, Response

from MCP_test import (
    search_by_claims,
    compute_claim_verdict,
    MAX_RESULTS_PER_SOURCE,
)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Stdout capture — intercepts print() calls from MCP_test and pushes them
# into a queue so we can stream them to the browser as SSE events.
# ---------------------------------------------------------------------------
class QueueWriter(io.TextIOBase):
    """File-like object that sends every write() to a queue."""

    def __init__(self, q):
        self.q = q

    def write(self, text):
        text = text.strip()
        if text:
            self.q.put(text)
        return len(text)

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_file("index.html")


@app.route("/api/search", methods=["POST"])
def api_search():
    """Run the pipeline, streaming progress via Server-Sent Events."""
    data = request.get_json(force=True)
    user_query = data.get("query", "").strip()
    if not user_query:
        return jsonify({"error": "No query provided"}), 400

    progress_q = queue.Queue()
    result_holder = [None]  # mutable container for thread result
    error_holder = [None]

    def run_pipeline():
        # Redirect stdout so all print() calls in MCP_test go to our queue
        old_stdout = sys.stdout
        sys.stdout = QueueWriter(progress_q)
        try:
            claim_results = search_by_claims(
                user_query, max_results_per_source=MAX_RESULTS_PER_SOURCE
            )
            result_holder[0] = claim_results
        except Exception as e:
            error_holder[0] = str(e)
        finally:
            sys.stdout = old_stdout
            progress_q.put(None)  # sentinel: pipeline done

    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()

    def generate():
        while True:
            try:
                msg = progress_q.get(timeout=120)
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'progress', 'message': 'Still working...'})}\n\n"
                continue

            if msg is None:
                break  # pipeline finished
            yield f"data: {json.dumps({'type': 'progress', 'message': msg})}\n\n"

        # Send final result or error
        if error_holder[0]:
            yield f"data: {json.dumps({'type': 'error', 'message': error_holder[0]})}\n\n"
        else:
            claims_data = _build_response(result_holder[0])
            yield f"data: {json.dumps({'type': 'result', 'claims': claims_data})}\n\n"

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _build_response(claim_results):
    """Convert pipeline output into JSON-serialisable claims list."""
    claims_data = []
    for claim_data in claim_results:
        papers = claim_data["results"].get("combined", [])
        verdict = compute_claim_verdict(claim_data)
        ct = verdict["claim_totals"]

        claim_info = {
            "claim_text": claim_data["claim_text"],
            "keywords": claim_data["keywords"],
            "verdict": {
                "agree": ct["agree"],
                "disagree": ct["disagree"],
                "missing": ct["missing"],
                "agreement_rate": round(ct["agreement_rate"], 3),
                "disagreement_rate": round(ct["disagreement_rate"], 3),
                "coverage_rate": round(ct["coverage_rate"], 3),
                "consistency_rate": round(ct["consistency_rate"], 3),
                "validity_score": round(ct["validity_score"], 3),
            },
            "papers": [],
        }

        for paper in papers[:10]:
            neg = paper.get("negation_scores", {})
            claim_info["papers"].append({
                "title": paper.get("title", ""),
                "authors": paper.get("authors", [])[:5],
                "source": paper.get("source", "unknown"),
                "similarity": round(paper.get("best_similarity", 0), 3),
                "abstract": (paper.get("abstract") or "")[:500],
                "pdf_url": paper.get("pdf_url", ""),
                "top_sentences": [
                    {"text": s[:300], "score": round(sc, 3)}
                    for s, sc in paper.get("top_sentences", [])[:3]
                ],
                "negation": {
                    "agree": neg.get("agree", 0),
                    "disagree": neg.get("disagree", 0),
                    "validity_score": round(neg.get("validity_score", 0), 3),
                } if neg else {},
            })

        claims_data.append(claim_info)

    return claims_data


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("LLM Academic Claim Verifier")
    print("Open http://localhost:5000 in your browser\n")
    app.run(debug=True, port=5000)

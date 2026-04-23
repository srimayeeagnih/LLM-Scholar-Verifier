"""
FAISS-backed semantic query cache for the Academic Claim Verifier.

On a cache hit (cosine similarity >= threshold), returns the stored pipeline
result directly — skipping all API calls, PDF downloads, and scoring.

Storage (both files live in backend/cache/):
    faiss_index.bin  — FAISS IndexFlatIP over L2-normalised sentence embeddings
    metadata.pkl     — parallel list of {query, result} dicts matching index IDs

Embedding model: all-MiniLM-L6-v2 (384-dim, ~80 MB, runs fully on CPU)
"""
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
INDEX_PATH = os.path.join(CACHE_DIR, "faiss_index.bin")
META_PATH  = os.path.join(CACHE_DIR, "metadata.pkl")

EMBEDDING_DIM       = 384   # all-MiniLM-L6-v2 output dimension
SIMILARITY_THRESHOLD = 0.90  # cosine similarity required to count as a cache hit


class FAISSQueryCache:
    """Singleton-safe semantic cache. Instantiate once at app startup."""

    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

        # Load embedding model once — stays in memory for the lifetime of the process
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

        if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH):
            self._index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "rb") as f:
                self._metadata: list[dict] = pickle.load(f)
            print(f"[cache] Loaded {self._index.ntotal} cached queries from disk.")
        else:
            self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self._metadata = []
            print("[cache] No cache found — starting fresh.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, threshold: float = SIMILARITY_THRESHOLD):
        """Return (cached_result, similarity_score) on hit, or None on miss."""
        if self._index.ntotal == 0:
            return None

        vec = self._embed(query)
        distances, indices = self._index.search(vec, k=1)

        score = float(distances[0][0])
        idx   = int(indices[0][0])

        if score >= threshold and idx < len(self._metadata):
            return self._metadata[idx]["result"], score

        return None

    def store(self, query: str, result: list) -> None:
        """Add a new query+result pair to the index and persist to disk."""
        vec = self._embed(query)
        self._index.add(vec)
        self._metadata.append({"query": query, "result": result})
        self._persist()

    def size(self) -> int:
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> np.ndarray:
        """Embed text and L2-normalise so inner product == cosine similarity."""
        vec = self._model.encode([text], normalize_embeddings=True)
        return vec.astype("float32")

    def _persist(self) -> None:
        faiss.write_index(self._index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self._metadata, f)

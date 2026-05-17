import json
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import numpy as np

# Deterministic mapping from catalog category strings to single-letter codes.
# Multi-category items produce comma-joined codes e.g. "P,C" or "B,S".
CATEGORY_TO_CODE: Dict[str, str] = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

def resolve_test_type(keys: List[str]) -> str:
    """Convert a list of catalog category strings to a comma-joined type code string."""
    codes = [CATEGORY_TO_CODE[k] for k in keys if k in CATEGORY_TO_CODE]
    return ",".join(codes) if codes else "K"


class CatalogRetriever:
    def __init__(self, data_path: str):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f, strict=False)

        self.documents = []
        tokenized_docs = []
        for item in self.data:
            name = item.get("name", "")
            desc = item.get("description", "")
            levels = item.get("job_levels_raw", "")
            keys = ", ".join(item.get("keys", []))
            doc = f"{name} {desc} {levels} {keys}"
            self.documents.append(doc)
            tokenized_docs.append(doc.lower().split())

        # TF-IDF for cosine similarity scoring
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

        # BM25 for better term-frequency saturation on short/vague queries
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        top_k = min(top_k, 10)  # hard cap — evaluator allows max 10 recommendations

        # TF-IDF scores (cosine similarity, 0–1)
        query_vec = self.vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # BM25 scores — normalise to 0–1 range for fair combination
        tokenized_query = query.lower().split()
        bm25_raw = np.array(self.bm25.get_scores(tokenized_query))
        bm25_max = bm25_raw.max()
        bm25_scores = bm25_raw / bm25_max if bm25_max > 0 else bm25_raw

        # Hybrid: equal-weight average of both signals
        hybrid_scores = 0.5 * tfidf_scores + 0.5 * bm25_scores

        top_indices = hybrid_scores.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if hybrid_scores[idx] > 0.01:  # relaxed threshold — BM25 lowers raw scores
                results.append(self.data[idx])
        return results

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        for item in self.data:
            if item.get("name", "").lower() == name.lower():
                return item
        return None

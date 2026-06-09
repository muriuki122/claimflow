import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
from scipy import sparse
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class MLArtifacts:
    vectorizer: TfidfVectorizer
    anomaly_model: IsolationForest
    neighbors_model: NearestNeighbors
    train_matrix: Any
    train_meta: List[Dict[str, Any]]
    score_mean: float
    score_std: float


class DocumentMLValidator:
    """Trainable ML validator for document quality and consistency."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.artifacts: Optional[MLArtifacts] = None
        self.load()

    @staticmethod
    def _safe_text(text: str) -> str:
        return (text or "").strip()

    @staticmethod
    def _meta_features(text: str, extracted_data: Dict[str, Any], requirements_status: Dict[str, Any]) -> np.ndarray:
        t = text or ""
        words = t.split()
        chars = max(len(t), 1)
        digits = sum(ch.isdigit() for ch in t)
        uppers = sum(ch.isupper() for ch in t)
        extracted_present = sum(1 for v in (extracted_data or {}).values() if v)
        req_found = sum(1 for v in (requirements_status or {}).values() if (v or {}).get("found"))
        return np.array([
            len(words),
            chars,
            digits / chars,
            uppers / chars,
            extracted_present,
            req_found,
        ], dtype=float)

    def _build_matrix(self, texts: List[str], metas: List[np.ndarray]):
        X_text = self.artifacts.vectorizer.transform(texts) if self.artifacts else None
        if X_text is None:
            raise RuntimeError("Vectorizer not initialized")
        X_meta = sparse.csr_matrix(np.vstack(metas))
        return sparse.hstack([X_text, X_meta], format="csr")

    def train(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        cleaned = []
        for r in records:
            txt = self._safe_text(r.get("text", ""))
            if len(txt) < 30:
                continue
            cleaned.append(r)

        if len(cleaned) < 3:
            raise ValueError("Need at least 3 usable documents to train ML validator")

        texts = [self._safe_text(r.get("text", "")) for r in cleaned]
        metas = [self._meta_features(t, r.get("extracted_data", {}), r.get("requirements_status", {})) for t, r in zip(texts, cleaned)]

        vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=1)
        X_text = vectorizer.fit_transform(texts)
        X_meta = sparse.csr_matrix(np.vstack(metas))
        X = sparse.hstack([X_text, X_meta], format="csr")

        anomaly_model = IsolationForest(
            n_estimators=300,
            contamination=0.20,
            random_state=42,
        )
        anomaly_model.fit(X)

        neighbors_model = NearestNeighbors(n_neighbors=min(5, len(cleaned)), metric="cosine")
        neighbors_model.fit(X_text)

        train_scores = anomaly_model.score_samples(X)

        self.artifacts = MLArtifacts(
            vectorizer=vectorizer,
            anomaly_model=anomaly_model,
            neighbors_model=neighbors_model,
            train_matrix=X_text,
            train_meta=cleaned,
            score_mean=float(np.mean(train_scores)),
            score_std=float(np.std(train_scores) + 1e-9),
        )
        self.save()

        return {
            "trained": True,
            "train_docs": len(cleaned),
            "score_mean": self.artifacts.score_mean,
            "score_std": self.artifacts.score_std,
        }

    def predict(self, text: str, extracted_data: Dict[str, Any], requirements_status: Dict[str, Any]) -> Dict[str, Any]:
        if not self.artifacts:
            return {"available": False, "reason": "model_not_trained"}

        text = self._safe_text(text)
        meta = self._meta_features(text, extracted_data, requirements_status)

        X_text = self.artifacts.vectorizer.transform([text])
        X_meta = sparse.csr_matrix(meta.reshape(1, -1))
        X = sparse.hstack([X_text, X_meta], format="csr")

        raw_score = float(self.artifacts.anomaly_model.score_samples(X)[0])
        z = (raw_score - self.artifacts.score_mean) / self.artifacts.score_std
        ml_quality_score = max(0.0, min(100.0, 50.0 + 20.0 * z))
        is_anomalous = bool(self.artifacts.anomaly_model.predict(X)[0] == -1)

        dists, idxs = self.artifacts.neighbors_model.kneighbors(X_text)
        neighbors = []
        for dist, idx in zip(dists[0], idxs[0]):
            rec = self.artifacts.train_meta[int(idx)]
            neighbors.append({
                "filename": rec.get("filename"),
                "similarity": float(1.0 - dist),
                "is_compliant": bool(rec.get("is_compliant", False)),
                "confidence_score": float(rec.get("confidence_score", 0)),
            })

        if neighbors:
            sims = np.array([n["similarity"] for n in neighbors], dtype=float)
            comps = np.array([1.0 if n["is_compliant"] else 0.0 for n in neighbors], dtype=float)
            weighted = float((sims * comps).sum() / max(sims.sum(), 1e-9))
        else:
            weighted = 0.0

        return {
            "available": True,
            "ml_quality_score": round(ml_quality_score, 2),
            "anomaly_flag": is_anomalous,
            "neighbor_compliance_probability": round(weighted, 4),
            "nearest_neighbors": neighbors,
        }

    def save(self):
        if not self.artifacts:
            return
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.artifacts, self.model_path)

    def load(self):
        if os.path.exists(self.model_path):
            self.artifacts = joblib.load(self.model_path)


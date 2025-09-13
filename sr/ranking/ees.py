# sr/ranking/ees.py
from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sr.config.schema import Signals

class EESModel:
    def __init__(self):
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.clf = SGDClassifier(loss="log_loss", random_state=17)
        self.fitted = False

    @staticmethod
    def features(sig: Signals) -> np.ndarray:
        return np.array([
            sig.pi_hits_title, sig.pi_hits_abstract, sig.tfidf_cos, sig.embed_cos,
            sig.design_prior, sig.recency_scaled, 1.0 if sig.abstract_missing else 0.0
        ], dtype=np.float32)

    def fit_epoch(self, pos: List[Signals], neg: List[Signals]):
        if not pos or not neg:
            return
        X = np.vstack([self.features(s) for s in (pos + neg)])
        y = np.array([1]*len(pos) + [0]*len(neg), dtype=np.int32)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.clf.partial_fit(Xs, y, classes=np.array([0,1]))
        self.fitted = True

    def predict(self, sigs: Dict[str, Signals]) -> Dict[str, float]:
        if not self.fitted:
            return {k: 0.5 for k in sigs.keys()}
        keys = list(sigs.keys())
        X = np.vstack([self.features(sigs[k]) for k in keys])
        Xs = self.scaler.transform(X)
        p = self.clf.predict_proba(Xs)[:,1]
        return {keys[i]: float(p[i]) for i in range(len(keys))}

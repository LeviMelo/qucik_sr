from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

PRIMARY_TYPES = {"Randomized Controlled Trial","Clinical Trial","Cohort Studies","Case-Control Studies","Observational Study","Controlled Clinical Trial","Prospective Studies"}
NON_PRIMARY_TYPES = {"Review","Meta-Analysis","Editorial","Letter","Comment","News","Case Reports","Guideline"}

def featurize_row(sig, doc) -> np.ndarray:
    # Continuous features + simple one-hots
    f = [
        sig.sem_intent, sig.sem_seed, sig.graph_ppr_pct/100.0, sig.graph_links_frac,
        sig.year_scaled,
        1.0 if sig.abstract_len_bin=="none" else 0.0,
        1.0 if sig.abstract_len_bin=="short" else 0.0,
        1.0 if sig.abstract_len_bin=="normal" else 0.0,
        1.0 if sig.abstract_len_bin=="long" else 0.0,
        1.0 if (set(doc.pub_types) & PRIMARY_TYPES) else 0.0,
        1.0 if (set(doc.pub_types) & NON_PRIMARY_TYPES) else 0.0,
    ]
    return np.array(f, dtype="float32")

class OnlineRegressor:
    def __init__(self):
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1000, tol=1e-3, random_state=17)
        self._fitted = False

    def fit_bootstrap(self, X: np.ndarray, y: np.ndarray):
        # Scale then partial_fit
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        self.clf.partial_fit(Xs, y, classes=np.array([0,1]))
        self._fitted = True

    def partial_update(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.transform(X)
        self.clf.partial_fit(Xs, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        p = self.clf.predict_proba(Xs)[:,1]
        return p

"""Lightweight input-distribution drift detection for the served GNN (PRD 7:
"a customer's topology changes as they ship new agents/tools, so the GNN's input
distribution will drift -- worth a lightweight model-drift check ... before this
becomes a customer-facing reliability problem for your own product").

Population Stability Index (PSI) per feature, computed against quantile bins fixed
at training time -- a standard, well-understood drift metric that needs no extra
dependency (Evidently AI or similar was the PRD's suggestion, but pulling in a full
observability framework for one metric isn't "lightweight"; PSI over numpy is).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Conventional PSI bands: <0.1 no meaningful drift, 0.1-0.2 moderate, >0.2 significant.
DRIFT_THRESHOLD = 0.2

_EPSILON = 1e-6  # avoids log(0)/div-by-0 for an empty bin


def compute_reference(features: np.ndarray, feature_names: list[str], n_bins: int = 10) -> dict:
    """Quantile-binned reference distribution per feature, computed once from the
    training feature matrix (shape [n_samples, n_features]) and persisted alongside
    the trained model -- serving never needs the raw training data, only this."""
    reference: dict = {}
    for i, name in enumerate(feature_names):
        column = features[:, i]
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(column, quantiles))
        if len(edges) < 2:
            # A feature that's constant across all of training (quantiles collapse
            # to one point) still needs two distinct edges for np.histogram below.
            # The +1.0 width is arbitrary but harmless: every training value falls
            # in this single bin regardless, and compute_drift bins `observed` into
            # these same edges, so a later non-constant value just lands outside it.
            edges = np.array([column.min(), column.min() + 1.0])
        counts, _ = np.histogram(column, bins=edges)
        proportions = counts / max(counts.sum(), 1)
        reference[name] = {"bin_edges": edges.tolist(), "bin_proportions": proportions.tolist()}
    return reference


def compute_drift(reference: dict, observed: np.ndarray, feature_names: list[str]) -> dict[str, float]:
    """PSI per feature: observed values binned into the *reference's* fixed edges,
    so a shift in the observed distribution shows up as bin proportions diverging
    from what training saw -- not a re-estimate that could hide the shift."""
    scores: dict[str, float] = {}
    for i, name in enumerate(feature_names):
        ref = reference[name]
        edges = np.array(ref["bin_edges"])
        ref_props = np.array(ref["bin_proportions"])
        column = observed[:, i]
        counts, _ = np.histogram(column, bins=edges)
        obs_props = counts / max(counts.sum(), 1)
        ref_safe = np.clip(ref_props, _EPSILON, None)
        obs_safe = np.clip(obs_props, _EPSILON, None)
        scores[name] = float(np.sum((obs_safe - ref_safe) * np.log(obs_safe / ref_safe)))
    return scores


def save_reference(reference: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(reference), encoding="utf-8")


def load_reference(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

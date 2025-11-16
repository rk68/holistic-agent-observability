from __future__ import annotations

import logging
import math
from functools import lru_cache
from os import getenv
from typing import Any, Dict, Iterable, List, Tuple

from sentence_transformers import CrossEncoder

logger = logging.getLogger("uvicorn.error")

_DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-small"


@lru_cache(maxsize=1)
def _load_model() -> CrossEncoder:
    model_name = getenv("TRACE_METRICS_MODEL", _DEFAULT_MODEL)
    logger.info("[TraceMetrics] Loading cross encoder model %s", model_name)
    return CrossEncoder(model_name, device="cpu")


def _softmax(logits: Iterable[float]) -> List[float]:
    values = list(float(item) for item in logits)
    if not values:
        return []
    max_logit = max(values)
    exp_values = [math.exp(v - max_logit) for v in values]
    total = sum(exp_values)
    if total == 0:
        return [0.0 for _ in exp_values]
    return [val / total for val in exp_values]


def score_pairs(pairs: List[Tuple[str, str]]) -> List[Dict[str, float]]:
    """Return NLI probabilities for each (premise, hypothesis) pair.

    The returned dict contains contradiction/neutral/entailment probabilities.
    """

    if not pairs:
        return []

    try:
        model = _load_model()
        raw_scores = model.predict(pairs)
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("[TraceMetrics] Failed scoring %d pairs: %r", len(pairs), exc)
        return []

    results: List[Dict[str, float]] = []
    for entry in raw_scores:
        if isinstance(entry, (list, tuple)):
            probs = _softmax(entry)
        elif hasattr(entry, "__len__") and not isinstance(entry, (str, bytes)):
            # Duck-typing to support numpy arrays and similar tensor outputs
            probs = _softmax(list(entry))
        else:
            # Some models may emit a single logit; treat it as entailment probability
            scalar: float
            try:
                scalar = float(entry)
            except TypeError:
                scalar = float(entry.item()) if hasattr(entry, "item") else 0.0
            scalar = max(0.0, min(1.0, scalar))
            probs = [0.0, 1.0 - scalar, scalar]
        if len(probs) == 3:
            contradiction, neutral, entailment = probs
        else:
            # Best-effort fallback mapping
            entailment = probs[-1]
            contradiction = probs[0] if probs else 0.0
            neutral = 1.0 - min(1.0, contradiction + entailment)
        results.append(
            {
                "contradiction": float(max(0.0, min(1.0, contradiction))),
                "neutral": float(max(0.0, min(1.0, neutral))),
                "entailment": float(max(0.0, min(1.0, entailment))),
            }
        )
    return results

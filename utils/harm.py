from typing import Any, Dict, List

import dill
import pandas as pd


def load_harm_detector(path: str):
    with open(path, "rb") as f:
        return dill.load(f)


def harm_proba(detector, text: str) -> float:
    return float(detector.predict_proba(pd.DataFrame({"text": [text]}))[:, 1][0])


def harm_proba_batch(detector, texts: List[str]) -> List[float]:
    if not texts:
        return []
    probs = detector.predict_proba(pd.DataFrame({"text": texts}))[:, 1]
    return [float(p) for p in probs]


def is_harmful_text(detector, text: str, threshold: float = 0.75) -> bool:
    return harm_proba(detector, text) > threshold


def is_harmful_cached(
    cache: Dict[Any, bool],
    key: Any,
    detector,
    text: str,
    threshold: float = 0.75,
) -> bool:
    if key in cache:
        return cache[key]
    verdict = is_harmful_text(detector, text, threshold=threshold)
    cache[key] = verdict
    return verdict


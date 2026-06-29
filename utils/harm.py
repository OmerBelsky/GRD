from typing import Any, Dict, List, Optional

import dill
import pandas as pd


def load_harm_detector(path: str):
    with open(path, "rb") as f:
        try:
            return dill.load(f)
        except RuntimeError as exc:
            message = str(exc)
            if "torchvision::nms" not in message:
                raise

            # Some serialized detectors reference transformers modules that try
            # to import torchvision at unpickle time. If torchvision is present
            # but broken in the environment, force Transformers to treat it as
            # unavailable and retry the same load.
            try:
                import transformers.utils.import_utils as import_utils

                import_utils._torchvision_available = False
                import_utils._torchvision_version = "N/A"
            except Exception:
                raise

            f.seek(0)
            return dill.load(f)


def harm_proba(detector, text: str) -> float:
    return float(detector.predict_proba(pd.DataFrame({"text": [text]}))[:, 1][0])


def harm_proba_cached(
    cache: Dict[Any, float],
    key: Any,
    detector,
    text: str,
) -> float:
    if key in cache:
        return cache[key]
    score = harm_proba(detector, text)
    cache[key] = score
    return score


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
    score_cache: Optional[Dict[Any, float]] = None,
) -> bool:
    if key in cache:
        return cache[key]
    if score_cache is not None:
        score = harm_proba_cached(score_cache, key, detector, text)
    else:
        score = harm_proba(detector, text)
    verdict = score > threshold
    cache[key] = verdict
    return verdict


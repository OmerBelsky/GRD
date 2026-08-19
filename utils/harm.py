import os
from typing import Any, Dict, Iterable, List, Optional

import dill
import numpy as np
import pandas as pd
import torch

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - transformers is optional at import time
    AutoModel = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]


class QwenGuardDetector:
    """Adapter for Qwen3Guard-Stream that exposes predict_proba() compatibly."""

    def __init__(self, model_name: str, hf_token: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.device = device or ("cuda:1" if torch.cuda.is_available() else "cpu")

        if AutoTokenizer is None or AutoModel is None:
            raise ImportError("transformers is required to load the Qwen Guard detector")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=self.hf_token,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            token=self.hf_token,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        ).to(self.device).eval()

    @staticmethod
    def _risk_to_harm_score(risk_level: str, risk_prob: float) -> float:
        risk_level = str(risk_level).strip()
        risk_prob = float(risk_prob)
        if risk_level == "Unsafe":
            return risk_prob
        if risk_level == "Safe":
            return 1.0 - risk_prob
        if risk_level == "Controversial":
            return 0.5
        return max(0.0, min(1.0, risk_prob))

    def _single_proba(self, text: str) -> np.ndarray:
        formatted = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": str(text)}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        inputs = self.tokenizer(formatted, return_tensors="pt")
        token_ids = inputs["input_ids"][0].to(self.device)

        stream_state = None
        try:
            result, stream_state = self.model.stream_moderate_from_ids(
                token_ids,
                role="user",
                stream_state=None,
            )
            risk_level = result["risk_level"][-1]
            risk_prob = float(result["risk_prob"][-1])
            harm_score = self._risk_to_harm_score(risk_level, risk_prob)
            safe_score = 1.0 - harm_score
            return np.array([safe_score, harm_score], dtype=float)
        finally:
            if stream_state is not None:
                self.model.close_stream(stream_state)

    def predict_proba(self, df: Any) -> np.ndarray:
        if isinstance(df, pd.DataFrame):
            texts = df["text"].tolist()
        elif isinstance(df, (list, tuple, np.ndarray)):
            texts = [str(item) for item in df]
        else:
            texts = [str(df)]

        probs = np.vstack([self._single_proba(text) for text in texts])
        return probs

    def __call__(self, payload: Any) -> np.ndarray:
        if isinstance(payload, pd.DataFrame):
            return self.predict_proba(payload)
        if isinstance(payload, str):
            return self.predict_proba(pd.DataFrame({"text": [payload]}))
        if isinstance(payload, Iterable) and not isinstance(payload, (bytes, str)):
            items = list(payload)
            return self.predict_proba(pd.DataFrame({"text": items}))
        return self.predict_proba(pd.DataFrame({"text": [payload]}))


def _extract_harm_probability(probabilities):
    arr = np.asarray(probabilities, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if arr.ndim == 1:
        if arr.size == 1:
            return float(arr[0])
        if arr.size == 2:
            return float(arr[1])
        if arr.size >= 3:
            # Qwen3Guard-style 3-way outputs are usually ordered as
            # [safe, harmful, other] or a close variant; the harmful class is
            # the middle logit-probability slot in the common 3-way layout.
            return float(arr[1])
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return float(arr[:, 0])
        if arr.shape[1] == 2:
            return arr[:, 1]
        if arr.shape[1] >= 3:
            return arr[:, 1]
    return float(arr)


def load_harm_detector(path: str):
    if path is None:
        raise ValueError("detector path must not be None")

    if not os.path.exists(path):
        normalized = str(path).strip()
        if "Qwen" in normalized or "qwen" in normalized or "Qwen3Guard" in normalized:
            return QwenGuardDetector(normalized)

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
    if hasattr(detector, "predict_proba"):
        probabilities = detector.predict_proba(pd.DataFrame({"text": [text]}))
        return float(_extract_harm_probability(probabilities)[0])

    if hasattr(detector, "__call__"):
        try:
            probabilities = detector(pd.DataFrame({"text": [text]}))
        except TypeError:
            probabilities = detector([text])
        return float(_extract_harm_probability(probabilities)[0])

    raise TypeError(f"Unsupported harm detector type: {type(detector)!r}")


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

    if hasattr(detector, "predict_proba"):
        probabilities = detector.predict_proba(pd.DataFrame({"text": texts}))
        extracted = _extract_harm_probability(probabilities)
        return [float(p) for p in np.asarray(extracted, dtype=float)]

    if hasattr(detector, "__call__"):
        try:
            probabilities = detector(pd.DataFrame({"text": texts}))
        except TypeError:
            probabilities = detector(texts)
        extracted = _extract_harm_probability(probabilities)
        return [float(p) for p in np.asarray(extracted, dtype=float)]

    raise TypeError(f"Unsupported harm detector type: {type(detector)!r}")


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


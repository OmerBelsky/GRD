import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DeviceMapType = Union[str, Dict[str, Union[int, str, torch.device]]]


def resolve_hf_token(hf_token: Optional[str]) -> Optional[str]:
    return hf_token or os.getenv("HF_TOKEN")


def resolve_target_device(device: Optional[str]) -> Optional[str]:
    # GRD_MODEL_DEVICE can force a single-device placement for all loaders.
    env_device = os.getenv("GRD_MODEL_DEVICE")
    if device:
        return device
    if env_device:
        return env_device
    return None


def resolve_device_map(device_map: Optional[DeviceMapType]) -> DeviceMapType:
    if device_map is not None:
        return device_map
    return os.getenv("GRD_DEVICE_MAP", "auto")


def seed_torch(seed: Optional[int]) -> None:
    if seed is None:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name: str, hf_token: Optional[str] = None) -> AutoTokenizer:
    token = resolve_hf_token(hf_token)
    return AutoTokenizer.from_pretrained(model_name, token=token)


def load_model_auto(
    model_name: str,
    hf_token: Optional[str] = None,
    *,
    dtype="auto",
    low_cpu_mem_usage: bool = True,
    device: Optional[str] = None,
    device_map: Optional[DeviceMapType] = None,
    torch_dtype=None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    token = resolve_hf_token(hf_token)
    target_device = resolve_target_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    resolved_dtype = dtype if torch_dtype is None else torch_dtype

    model_kwargs: Dict[str, Any] = {
        "token": token,
        "dtype": resolved_dtype,
        "low_cpu_mem_usage": low_cpu_mem_usage,
    }

    if target_device is None:
        model_kwargs["device_map"] = resolve_device_map(device_map)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if target_device is not None:
        model = model.to(target_device)
    model.eval()
    return tokenizer, model


def load_llm_auto(
    model_name: str,
    hf_token: Optional[str] = None,
    *,
    dtype="auto",
    low_cpu_mem_usage: bool = True,
    device: Optional[str] = None,
    device_map: Optional[DeviceMapType] = None,
    torch_dtype=None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, torch.device]:
    tokenizer, model = load_model_auto(
        model_name,
        hf_token,
        dtype=dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        device=device,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return tokenizer, model, model.device


def load_llm_to_device(
    model_name: str,
    hf_token: Optional[str] = None,
    *,
    device: Optional[str] = None,
    dtype=None,
    torch_dtype=None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, str]:
    resolved_device = resolve_target_device(device) or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_model_auto(
        model_name,
        hf_token,
        dtype=dtype if dtype is not None else "auto",
        low_cpu_mem_usage=True,
        device=resolved_device,
        device_map=None,
        torch_dtype=torch_dtype,
    )
    return tokenizer, model, resolved_device

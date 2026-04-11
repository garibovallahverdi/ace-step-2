"""Model initialization helpers for RunPod serverless execution."""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional, Tuple

from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.model_downloader import (
    ensure_dit_model,
    ensure_lm_model,
    get_checkpoints_dir,
)

_init_lock = threading.Lock()
_dit_handler: Optional[AceStepHandler] = None
_llm_handler: Optional[LLMHandler] = None


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _should_init_llm(input_data: Dict[str, Any]) -> bool:
    env_setting = os.getenv("ACESTEP_INIT_LLM", "auto").strip().lower()
    if env_setting in {"0", "false", "no"}:
        return False
    if env_setting in {"1", "true", "yes"}:
        return True
    return bool(
        input_data.get("init_llm")
        or input_data.get("thinking")
        or input_data.get("use_cot_caption")
        or input_data.get("use_cot_language")
        or input_data.get("use_cot_metas")
    )


def ensure_handlers(
    input_data: Dict[str, Any],
) -> Tuple[AceStepHandler, Optional[LLMHandler]]:
    """Ensure DiT/LM handlers are initialized and ready."""
    global _dit_handler, _llm_handler
    model_name = str(
        input_data.get("model")
        or os.getenv("ACESTEP_CONFIG_PATH", "acestep-v15-xl-turbo")
    ).strip()
    prefer_source = os.getenv("ACESTEP_DOWNLOAD_SOURCE") or None
    checkpoint_dir = str(get_checkpoints_dir())

    with _init_lock:
        if _dit_handler is None:
            _dit_handler = AceStepHandler()

        if _dit_handler.model is None:
            ok, msg = ensure_dit_model(
                model_name=model_name,
                checkpoints_dir=checkpoint_dir,
                prefer_source=prefer_source,
            )
            if not ok:
                raise RuntimeError(f"DiT model unavailable: {msg}")

            status, success = _dit_handler.initialize_service(
                project_root=_get_project_root(),
                config_path=model_name,
                device=os.getenv("ACESTEP_DEVICE", "auto"),
                use_flash_attention=_env_bool("ACESTEP_FLASH_ATTN", False),
                compile_model=_env_bool("TORCH_COMPILE", False),
                offload_to_cpu=_env_bool("CPU_OFFLOAD", False),
                offload_dit_to_cpu=_env_bool("ACESTEP_OFFLOAD_DIT_TO_CPU", False),
                quantization=os.getenv("ACESTEP_QUANTIZATION") or None,
                prefer_source=prefer_source,
            )
            if not success:
                raise RuntimeError(status)

        llm_handler: Optional[LLMHandler] = None
        if _should_init_llm(input_data):
            if _llm_handler is None:
                _llm_handler = LLMHandler()
            llm_model = str(
                input_data.get("lm_model_path")
                or os.getenv("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
            ).strip()
            if not _llm_handler.llm_initialized:
                ok, msg = ensure_lm_model(
                    model_name=llm_model,
                    checkpoints_dir=checkpoint_dir,
                    prefer_source=prefer_source,
                )
                if not ok:
                    raise RuntimeError(f"LM model unavailable: {msg}")
                status, success = _llm_handler.initialize(
                    checkpoint_dir=checkpoint_dir,
                    lm_model_path=llm_model,
                    backend=str(
                        input_data.get("lm_backend")
                        or os.getenv("ACESTEP_LM_BACKEND", "vllm")
                    ).strip(),
                    device=_dit_handler.device,
                    offload_to_cpu=_env_bool("CPU_OFFLOAD", False),
                )
                if not success:
                    raise RuntimeError(status)
            llm_handler = _llm_handler

    return _dit_handler, llm_handler

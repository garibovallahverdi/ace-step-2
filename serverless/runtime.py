"""Runtime helpers for RunPod serverless ACE-Step generation."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger

from acestep.inference import GenerationConfig, GenerationParams, generate_music
from supabase_store import SupabaseStore
from serverless.models import ensure_handlers


if "TORCHAUDIO_USE_BACKEND" not in os.environ:
    os.environ["TORCHAUDIO_USE_BACKEND"] = "soundfile"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_generation_lock = threading.Lock()
_supabase_store = SupabaseStore.from_env()


def _coerce_int(value: Any, default: Optional[int]) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_params_config(
    input_data: Dict[str, Any],
) -> Tuple[GenerationParams, GenerationConfig, str]:
    prompt = str(input_data.get("prompt") or "")
    lyrics = str(input_data.get("lyrics") or "")
    audio_duration = input_data.get("audio_duration")
    audio_format = str(
        input_data.get("audio_format")
        or os.getenv("ACESTEP_AUDIO_FORMAT_DEFAULT", "flac")
    ).strip().lower()

    params = GenerationParams(
        caption=prompt,
        lyrics=lyrics,
        bpm=_coerce_int(input_data.get("bpm"), None),
        keyscale=str(input_data.get("key_scale") or ""),
        timesignature=str(input_data.get("time_signature") or ""),
        vocal_language=str(input_data.get("vocal_language") or "unknown"),
        duration=_coerce_float(audio_duration, -1.0),
        inference_steps=_coerce_int(input_data.get("inference_steps"), 8),
        guidance_scale=_coerce_float(input_data.get("guidance_scale"), 7.0),
        thinking=bool(input_data.get("thinking", False)),
        task_type=str(input_data.get("task_type") or "text2music"),
        use_cot_metas=bool(input_data.get("use_cot_metas", True)),
        use_cot_caption=bool(input_data.get("use_cot_caption", True)),
        use_cot_language=bool(input_data.get("use_cot_language", True)),
        use_constrained_decoding=bool(input_data.get("constrained_decoding", True)),
    )

    use_random_seed = bool(input_data.get("use_random_seed", True))
    seed_value = input_data.get("seed", None)
    seeds = None if use_random_seed else seed_value

    config = GenerationConfig(
        batch_size=_coerce_int(input_data.get("batch_size"), 1),
        allow_lm_batch=bool(input_data.get("allow_lm_batch", True)),
        use_random_seed=use_random_seed,
        seeds=seeds,
        audio_format=audio_format,
        constrained_decoding_debug=bool(
            input_data.get("constrained_decoding_debug", False)
        ),
    )

    output_dir = os.getenv("OUTPUT_DIR", "/tmp/outputs")
    os.makedirs(output_dir, exist_ok=True)
    return params, config, output_dir


def _upload_outputs(
    audio_paths: list[str],
    request_id: str,
    metadata: Dict[str, Any],
) -> list[Dict[str, Any]]:
    uploads: list[Dict[str, Any]] = []
    if not _supabase_store.enabled:
        return uploads
    for idx, audio_path in enumerate(audio_paths):
        if not audio_path:
            uploads.append(
                {
                    "enabled": True,
                    "uploaded": False,
                    "message": "Skipped Supabase upload: empty audio path.",
                    "error": "empty_audio_path",
                }
            )
            continue
        uploads.append(
            _supabase_store.persist_generated_audio_safe(
                audio_path=audio_path,
                generation_id=f"{request_id}_{idx}",
                metadata=metadata,
            )
        )
    return uploads


def _cleanup_after_run() -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def handle_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """RunPod handler: generates music and returns local paths + Supabase URLs."""
    request_id = str(event.get("id") or uuid.uuid4().hex)
    input_data = event.get("input", {}) if isinstance(event, dict) else {}

    try:
        dit_handler, llm_handler = ensure_handlers(input_data)
        params, config, output_dir = _build_params_config(input_data)
        with _generation_lock:
            result = generate_music(
                dit_handler=dit_handler,
                llm_handler=llm_handler,
                params=params,
                config=config,
                save_dir=output_dir,
            )

        if not result.success:
            return {"status": "failed", "error": result.error or result.status_message}

        audio_paths = [audio.get("path") for audio in result.audios if audio.get("path")]
        metadata = {
            "prompt": params.caption,
            "lyrics": params.lyrics,
            "bpm": params.bpm,
            "duration": params.duration,
            "keyscale": params.keyscale,
            "timesignature": params.timesignature,
            "audio_format": config.audio_format,
        }
        uploads = _upload_outputs(audio_paths, request_id, metadata)

        return {
            "status": "succeeded",
            "request_id": request_id,
            "audio_paths": audio_paths,
            "supabase_uploads": uploads,
            "generation_info": result.extra_outputs.get("time_costs", {}),
        }
    except Exception as exc:
        logger.exception("Serverless generation failed: {}", exc)
        return {"status": "failed", "error": str(exc)}
    finally:
        _cleanup_after_run()

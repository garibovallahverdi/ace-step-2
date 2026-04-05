import asyncio
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from acestep.pipeline_ace_step import ACEStepPipeline
from supabase_store import SupabaseStore

  
app = FastAPI(title="ACEStep Pipeline API")
model: Optional[ACEStepPipeline] = None
model_checkpoint_path: Optional[str] = None
generation_lock = asyncio.Lock()
supabase_store = SupabaseStore.from_env()


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


class ACEStepInput(BaseModel):
    checkpoint_path: Optional[str] = None
    output_path: Optional[str] = None
    audio_duration: float = 60.0
    prompt: str
    lyrics: str = ""
    infer_step: int = 60
    guidance_scale: float = 15.0
    scheduler_type: str = "euler"
    cfg_type: str = "apg"
    omega_scale: float = 10.0
    actual_seeds: Optional[List[int]] = None
    guidance_interval: float = 0.5
    guidance_interval_decay: float = 0.0
    min_guidance_scale: float = 3.0
    use_erg_tag: bool = True
    use_erg_lyric: bool = True
    use_erg_diffusion: bool = True
    oss_steps: Optional[List[int]] = None
    guidance_scale_text: float = 0.0
    guidance_scale_lyric: float = 0.0
    upload_to_supabase: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ACEStepOutput(BaseModel):
    status: str
    output_path: str
    message: str
    generation_id: str
    input_params: Dict[str, Any]
    supabase: Optional[Dict[str, Any]] = None


@app.on_event("startup")
async def startup_event() -> None:
    global model
    global model_checkpoint_path

    checkpoint_path = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    device_id = int(os.getenv("DEVICE_ID", "0"))
    bf16 = _env_bool("BF16", True)
    torch_compile = _env_bool("TORCH_COMPILE", False)
    cpu_offload = _env_bool("CPU_OFFLOAD", False)
    overlapped_decode = _env_bool("OVERLAPPED_DECODE", False)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    model = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
        cpu_offload=cpu_offload,
        overlapped_decode=overlapped_decode,
    )
    model.load_checkpoint(checkpoint_path)
    model_checkpoint_path = checkpoint_path
    logger.info(
        "Model initialized on startup with checkpoint_path='{}', bf16={}, torch_compile={}, cpu_offload={}",
        checkpoint_path,
        bf16,
        torch_compile,
        cpu_offload,
    )


@app.post("/generate", response_model=ACEStepOutput)
async def generate_audio(input_data: ACEStepInput) -> ACEStepOutput:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not initialized yet")

    if input_data.checkpoint_path and input_data.checkpoint_path != model_checkpoint_path:
        raise HTTPException(
            status_code=400,
            detail=(
                "Per-request checkpoint switching is disabled for production stability. "
                f"Current loaded checkpoint: '{model_checkpoint_path}'."
            ),
        )

    generation_id = uuid.uuid4().hex
    output_dir = os.getenv("OUTPUT_DIR", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = input_data.output_path or os.path.join(output_dir, f"{generation_id}.wav")

    try:
        async with generation_lock:
            result = model(
                audio_duration=input_data.audio_duration,
                prompt=input_data.prompt,
                lyrics=input_data.lyrics,
                infer_step=input_data.infer_step,
                guidance_scale=input_data.guidance_scale,
                scheduler_type=input_data.scheduler_type,
                cfg_type=input_data.cfg_type,
                omega_scale=input_data.omega_scale,
                manual_seeds=input_data.actual_seeds,
                guidance_interval=input_data.guidance_interval,
                guidance_interval_decay=input_data.guidance_interval_decay,
                min_guidance_scale=input_data.min_guidance_scale,
                use_erg_tag=input_data.use_erg_tag,
                use_erg_lyric=input_data.use_erg_lyric,
                use_erg_diffusion=input_data.use_erg_diffusion,
                oss_steps=input_data.oss_steps or [],
                guidance_scale_text=input_data.guidance_scale_text,
                guidance_scale_lyric=input_data.guidance_scale_lyric,
                save_path=output_path,
            )

        generated_paths = [item for item in result if isinstance(item, str)]
        input_params = next((item for item in result if isinstance(item, dict)), {})
        final_output_path = generated_paths[0] if generated_paths else output_path

        supabase_result: Optional[Dict[str, Any]] = None
        if input_data.upload_to_supabase:
            supabase_result = supabase_store.persist_generated_audio(
                audio_path=final_output_path,
                generation_id=generation_id,
                metadata={
                    **input_data.metadata,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "audio_duration": input_data.audio_duration,
                    "infer_step": input_data.infer_step,
                    "guidance_scale": input_data.guidance_scale,
                    "scheduler_type": input_data.scheduler_type,
                    "cfg_type": input_data.cfg_type,
                    "actual_seeds": input_data.actual_seeds,
                    "oss_steps": input_data.oss_steps,
                },
            )

        return ACEStepOutput(
            status="success",
            output_path=final_output_path,
            message="Audio generated successfully",
            generation_id=generation_id,
            input_params=input_params,
            supabase=supabase_result,
        )
    except Exception as exc:
        logger.exception("Generation failed: {}", exc)
        raise HTTPException(status_code=500, detail=f"Error generating audio: {exc}") from exc


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "checkpoint_path": model_checkpoint_path,
        "supabase_s3_enabled": supabase_store.enabled,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)

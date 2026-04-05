# Self Deploy Guide

## 1. Prepare the server
- Install Docker + Docker Compose plugin
- Install NVIDIA driver + NVIDIA Container Toolkit
- Validate GPU access:
  ```bash
  nvidia-smi
  docker run --rm --gpus all nvidia/cuda:12.6.0-runtime-ubuntu22.04 nvidia-smi
  ```

## 2. Configure environment
```bash
cp .env.example .env
```

Set at least:
- `S3_ACCESS_KEY`
- `S3_SECRET_KEY`
- `S3_ENDPOINT`
- `BUCKET_NAME` (default: `music`)
- `REGION_NAME` (default: `ap-southeast-2`)

Optional:
- `APP_MODE=api` (recommended for backend)
- `APP_MODE=gui` (if you want Gradio UI)

## 3. Prepare directories
```bash
mkdir -p checkpoints outputs logs
```

## 4. Build and run
```bash
docker compose up -d --build
```

## 5. Health check
```bash
curl http://localhost:8000/health
```

## 6. Test generation
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "cinematic ambient, emotional strings, soft piano",
    "lyrics": "",
    "audio_duration": 30,
    "infer_step": 27,
    "upload_to_supabase": true
  }'
```

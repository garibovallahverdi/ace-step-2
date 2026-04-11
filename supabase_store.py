import mimetypes
import os
import time
from typing import Any, Dict, Optional

import boto3
from botocore.client import Config
from botocore.exceptions import BotoCoreError, ClientError
from loguru import logger


class SupabaseStore:
    def __init__(
        self,
        access_key: Optional[str],
        secret_key: Optional[str],
        endpoint: Optional[str],
        bucket: str,
        region: str,
        signed_url_ttl_seconds: int,
    ) -> None:
        self.bucket = bucket
        self.region = region
        self.signed_url_ttl_seconds = signed_url_ttl_seconds
        self.enabled = bool(access_key and secret_key and endpoint and bucket and region)
        self.client = None

        if self.enabled:
            self.client = boto3.client(
                "s3",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                endpoint_url=endpoint,
                region_name=region,
                config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
            )
            logger.info(
                "Supabase S3 integration enabled with bucket='{}' and region='{}'",
                self.bucket,
                self.region,
            )
        else:
            logger.warning(
                "Supabase S3 integration is disabled. Set S3_ACCESS_KEY, S3_SECRET_KEY, "
                "S3_ENDPOINT, BUCKET_NAME and REGION_NAME to enable uploads."
            )

    @classmethod
    def from_env(cls) -> "SupabaseStore":
        return cls(
            access_key=os.getenv("S3_ACCESS_KEY"),
            secret_key=os.getenv("S3_SECRET_KEY"),
            endpoint=os.getenv("S3_ENDPOINT"),
            bucket=os.getenv("BUCKET_NAME", "music"),
            region=os.getenv("REGION_NAME", "ap-southeast-2"),
            signed_url_ttl_seconds=int(os.getenv("S3_SIGNED_URL_TTL_SECONDS", "3600")),
        )

    def persist_generated_audio(
        self,
        audio_path: str,
        generation_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "enabled": False,
                "uploaded": False,
                "message": "Supabase S3 is not configured, skipped upload.",
            }

        if not self.client:
            raise RuntimeError("Supabase S3 client was not initialized.")

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Generated audio file not found: {audio_path}")

        ext = os.path.splitext(audio_path)[1] or ".wav"
        object_path = f"tracks/{generation_id}{ext}"
        content_type = mimetypes.guess_type(audio_path)[0] or "audio/wav"

        try:
            self.client.upload_file(
                Filename=audio_path,
                Bucket=self.bucket,
                Key=object_path,
                ExtraArgs={"ContentType": content_type},
            )
            signed_url = self.client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.bucket, "Key": object_path},
                ExpiresIn=self.signed_url_ttl_seconds,
            )
        except (BotoCoreError, ClientError) as exc:
            logger.exception("Supabase S3 upload failed: {}", exc)
            raise RuntimeError(f"Supabase S3 upload failed: {exc}") from exc

        return {
            "enabled": True,
            "uploaded": True,
            "bucket": self.bucket,
            "storage_path": object_path,
            "signed_url": signed_url,
            "metadata": metadata or {},
            "message": "File uploaded to Supabase Storage (S3 API) successfully.",
        }

    def persist_generated_audio_safe(
        self,
        audio_path: str,
        generation_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        retries: int = 2,
        retry_backoff_seconds: float = 1.0,
    ) -> Dict[str, Any]:
        """Upload generated audio with retries, never raising errors.

        Args:
            audio_path: Path to the generated audio file.
            generation_id: Unique ID used for the object key.
            metadata: Optional metadata to include in the response.
            retries: Number of retries after the initial attempt.
            retry_backoff_seconds: Base backoff time in seconds between retries.

        Returns:
            A response dict describing upload success/failure. Exceptions are
            captured in the response rather than raised.
        """
        if not self.enabled:
            return {
                "enabled": False,
                "uploaded": False,
                "message": "Supabase S3 is not configured, skipped upload.",
                "error": None,
            }

        attempts = max(0, int(retries)) + 1
        last_error: Optional[str] = None
        for attempt in range(attempts):
            try:
                return self.persist_generated_audio(
                    audio_path=audio_path,
                    generation_id=generation_id,
                    metadata=metadata,
                )
            except Exception as exc:
                last_error = str(exc)
                logger.exception(
                    "Supabase S3 upload attempt {}/{} failed: {}",
                    attempt + 1,
                    attempts,
                    last_error,
                )
                if attempt < attempts - 1:
                    sleep_seconds = retry_backoff_seconds * (attempt + 1)
                    time.sleep(max(0.0, sleep_seconds))

        return {
            "enabled": True,
            "uploaded": False,
            "message": "Supabase S3 upload failed after retries.",
            "error": last_error,
            "metadata": metadata or {},
        }

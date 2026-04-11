"""Unit tests for SupabaseStore safe upload behavior."""

import unittest
from unittest import mock

from supabase_store import SupabaseStore


class SupabaseStoreSafeUploadTests(unittest.TestCase):
    """Behavior tests for SupabaseStore.persist_generated_audio_safe."""

    def test_safe_upload_returns_disabled_when_not_configured(self):
        """Disabled store should skip uploads without raising errors."""

        store = SupabaseStore(
            access_key=None,
            secret_key=None,
            endpoint=None,
            bucket="",
            region="",
            signed_url_ttl_seconds=3600,
        )

        result = store.persist_generated_audio_safe("missing.wav", "job-1")

        self.assertFalse(result["enabled"])
        self.assertFalse(result["uploaded"])

    def test_safe_upload_returns_error_when_upload_fails(self):
        """Safe upload should return an error response when retries fail."""

        store = SupabaseStore(
            access_key=None,
            secret_key=None,
            endpoint=None,
            bucket="",
            region="",
            signed_url_ttl_seconds=3600,
        )
        store.enabled = True
        store.persist_generated_audio = mock.Mock(side_effect=RuntimeError("boom"))

        result = store.persist_generated_audio_safe(
            "missing.wav",
            "job-2",
            retries=1,
            retry_backoff_seconds=0.0,
        )

        self.assertTrue(result["enabled"])
        self.assertFalse(result["uploaded"])
        self.assertIn("boom", result.get("error", ""))


if __name__ == "__main__":
    unittest.main()

"""RunPod Serverless entrypoint for ACE-Step generation."""

from __future__ import annotations

import runpod

from serverless.runtime import handle_event


runpod.serverless.start({"handler": handle_event})

"""Manual Gemma4 text-only dense MTP validation on CPU/Intel AMX.

This is intentionally separate from ``test_gemma4_mtp.py``, which targets the
CUDA graph/top-k path.  The CPU path currently supports Frozen-KV MTP in eager
mode with ``intel_amx`` attention and ``topk=1``.

Example:

    SGLANG_GEMMA4_CPU_TARGET=google/gemma-4-E2B-it \
    SGLANG_GEMMA4_CPU_ASSISTANT=google/gemma-4-E2B-it-assistant \
    SGLANG_GEMMA4_CPU_PROMPTS='["hello, who are you?"]' \
    python3 -m unittest test.manual.models.test_gemma4_mtp_cpu

Optional env vars:

* ``SGLANG_GEMMA4_CPU_TARGET``
* ``SGLANG_GEMMA4_CPU_ASSISTANT``
* ``SGLANG_GEMMA4_CPU_DTYPE`` (default: ``bfloat16``)
* ``SGLANG_GEMMA4_CPU_PROMPTS`` (JSON list of prompts)
* ``SGLANG_GEMMA4_CPU_MAX_TOKENS`` (default: ``48``)
* ``SGLANG_GEMMA4_CPU_TIMEOUT`` (default: server launch timeout)
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from typing import Dict, List, Optional

import requests


TARGET_ENV = "SGLANG_GEMMA4_CPU_TARGET"
ASSISTANT_ENV = "SGLANG_GEMMA4_CPU_ASSISTANT"
DTYPE_ENV = "SGLANG_GEMMA4_CPU_DTYPE"
PROMPTS_ENV = "SGLANG_GEMMA4_CPU_PROMPTS"
MAX_TOKENS_ENV = "SGLANG_GEMMA4_CPU_MAX_TOKENS"
TIMEOUT_ENV = "SGLANG_GEMMA4_CPU_TIMEOUT"


def _env_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise unittest.SkipTest(f"{name} is required for manual CPU Gemma4 MTP test")
    return value


def _ensure_checkpoint(path: str, label: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    if not (p / "config.json").exists():
        raise FileNotFoundError(f"{label} checkpoint at {path} misses config.json")
    if not list(p.glob("*.safetensors")):
        raise FileNotFoundError(f"{label} checkpoint at {path} misses safetensors")


def _prompts() -> List[str]:
    raw = os.getenv(PROMPTS_ENV)
    if raw:
        prompts = json.loads(raw)
        if not isinstance(prompts, list) or not all(isinstance(x, str) for x in prompts):
            raise ValueError(f"{PROMPTS_ENV} must be a JSON list of strings")
        return prompts
    return [
        "hello, who are you?",
        "Write a short explanation of speculative decoding.",
    ]


def _server_env() -> Dict[str, str]:
    env = dict(os.environ)
    env["SGLANG_ENABLE_SPEC_V2"] = "0"
    env["SGLANG_USE_CPU_ENGINE"] = "1"
    return env


def _server_info(base_url: str) -> Dict:
    response = requests.get(base_url + "/server_info", timeout=30)
    response.raise_for_status()
    return response.json()


def _completion(base_url: str, prompt: str, temperature: float) -> str:
    response = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": int(os.getenv(MAX_TOKENS_ENV, "48")),
            },
        },
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict):
        return payload.get("text", "")
    raise AssertionError(f"unexpected /generate payload: {payload!r}")


class TestGemma4MTPCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.target_path = _env_required(TARGET_ENV)
        cls.assistant_path = _env_required(ASSISTANT_ENV)
        cls.dtype = os.getenv(DTYPE_ENV, "bfloat16")
        _ensure_checkpoint(cls.target_path, "target")
        _ensure_checkpoint(cls.assistant_path, "assistant")

        from sglang.test.test_utils import (
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            DEFAULT_URL_FOR_TEST,
            find_available_port,
        )

        default_port = int(DEFAULT_URL_FOR_TEST.rsplit(":", 1)[1])
        cls.base_url = f"http://127.0.0.1:{find_available_port(default_port)}"
        cls.timeout = int(os.getenv(TIMEOUT_ENV, str(DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)))

    @staticmethod
    def _stop_process(process) -> None:
        from sglang.srt.utils import kill_process_tree

        try:
            kill_process_tree(process.pid)
        except Exception:
            pass
        try:
            process.wait(timeout=30)
        except Exception:
            pass

    @classmethod
    def _common_args(cls) -> List[str]:
        return [
            "--device",
            "cpu",
            "--attention-backend",
            "intel_amx",
            "--dtype",
            cls.dtype,
            "--disable-cuda-graph",
        ]

    @classmethod
    def _mtp_args(cls) -> List[str]:
        return [
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-draft-model-path",
            cls.assistant_path,
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
        ] + cls._common_args()

    def _launch(self, other_args: List[str]):
        from sglang.test.test_utils import popen_launch_server

        return popen_launch_server(
            self.target_path,
            self.base_url,
            timeout=self.timeout,
            env=_server_env(),
            other_args=other_args,
            device="cpu",
        )

    def test_greedy_outputs_match_baseline_and_accepts_drafts(self) -> None:
        prompts = _prompts()
        baseline = self._launch(self._common_args())
        try:
            baseline_outputs = [
                _completion(self.base_url, prompt, temperature=0.0) for prompt in prompts
            ]
        finally:
            self._stop_process(baseline)

        mtp = self._launch(self._mtp_args())
        try:
            info = _server_info(self.base_url)
            self.assertEqual(info.get("speculative_algorithm"), "FROZEN_KV_MTP")
            self.assertEqual(info.get("speculative_eagle_topk"), 1)
            self.assertTrue(info.get("disable_cuda_graph"))
            mtp_outputs = [
                _completion(self.base_url, prompt, temperature=0.0) for prompt in prompts
            ]
            info = _server_info(self.base_url)
            internal_states = info.get("internal_states") or []
            avg_accept = (
                internal_states[0].get("avg_spec_accept_length")
                if internal_states
                else None
            )
        finally:
            self._stop_process(mtp)

        self.assertEqual(mtp_outputs, baseline_outputs)
        self.assertIsNotNone(avg_accept)
        self.assertGreater(float(avg_accept), 0.0)

    def test_sampling_request_runs_with_mtp(self) -> None:
        mtp = self._launch(self._mtp_args())
        try:
            info = _server_info(self.base_url)
            self.assertEqual(info.get("speculative_algorithm"), "FROZEN_KV_MTP")
            for prompt in _prompts()[:1]:
                output = _completion(self.base_url, prompt, temperature=0.7)
                self.assertTrue(output.strip())
        finally:
            self._stop_process(mtp)


if __name__ == "__main__":
    unittest.main()

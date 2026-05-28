"""
Base class for LLM inference providers.

Provides shared constructor, method delegation (chat -> chat_json ->
_extract), exponential backoff, and param mapping. Provider-specific
subclasses implement _build_payload, _request_json_with_retry, and
may override hooks like _extract_chat_text_and_usage or _post_init.
"""

import os
import time
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple


class BaseInference(ABC):
    """Abstract base for chat-completion inference providers.

    Subclasses must set:
        API_URL: str
        API_KEY_ENV: str  (environment variable name for the API key)

    Subclasses typically override:
        _build_headers() -> dict
        _build_payload(messages, sampling_params) -> dict
        _request_json_with_retry(payload) -> dict
        _post_init()  (extra init steps, e.g. rate-limiter setup)
        _extract_chat_text_and_usage(data) -> (str, dict)
    """

    API_URL: str = ""
    API_KEY_ENV: str = ""

    def __init__(
        self,
        model: str,
        api_key: str = None,
        max_retries: int = 3,
        base_delay: float = 10.0,
        max_delay: float = 60.0,
        **kwargs,
    ):
        self.model = model
        self.model_name = model
        self.api_key = api_key or os.environ.get(self.API_KEY_ENV)

        if not self.api_key:
            raise ValueError(
                f"API key required. Set {self.API_KEY_ENV} or pass api_key."
            )

        self.max_retries = min(3, max(0, int(max_retries)))
        self.base_delay = base_delay
        self.max_delay = max_delay
        kwargs.pop("enable_caching", None)
        self.default_kwargs = kwargs

        self.headers = self._build_headers()
        self._post_init()

    # ── hooks for subclasses ──────────────────────────────────────────

    def _post_init(self) -> None:
        """Hook called at the end of __init__."""

    def _build_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    # ── public interface ──────────────────────────────────────────────

    def chat(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> Tuple[str, Dict]:
        data = self.chat_json(messages, sampling_params)
        return self._extract_chat_text_and_usage(data)

    def chat_json(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = self._build_payload(messages, sampling_params)
        return self._request_json_with_retry(payload)

    # ── abstract / overridable ────────────────────────────────────────

    @abstractmethod
    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the JSON request body."""

    @abstractmethod
    def _request_json_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to API_URL with retry logic; return parsed JSON dict."""

    @staticmethod
    def _extract_chat_text_and_usage(data: Dict[str, Any]) -> Tuple[str, Dict]:
        """Extract (content, usage) from a successful chat completion response."""
        if "choices" not in data:
            return "", {}

        message = data["choices"][0]["message"]
        content = message.get("content")
        usage = data.get("usage", {})

        if isinstance(content, str):
            return content, usage
        if content is None:
            return "", usage
        return str(content), usage

    def _calculate_backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter: min(max_delay, base_delay * 2^attempt) * jitter."""
        delay = min(self.max_delay, self.base_delay * (2 ** attempt))
        jitter = random.uniform(0.75, 1.25)
        return delay * jitter

    # ── helpers for subclasses ────────────────────────────────────────

    @staticmethod
    def _get_param_mapping() -> Dict[str, str]:
        """Return the shared sampling-param -> API-key mapping.

        Subclasses can override to add provider-specific keys (e.g.
        parallel_tool_calls for OpenRouter).
        """
        return {
            "temperature": "temperature",
            "max_tokens": "max_tokens",
            "top_p": "top_p",
            "top_k": "top_k",
            "frequency_penalty": "frequency_penalty",
            "presence_penalty": "presence_penalty",
            "stop": "stop",
            "tools": "tools",
            "tool_choice": "tool_choice",
        }

    def _apply_param_mapping(
        self,
        payload: Dict[str, Any],
        sampling_params: Dict[str, Any],
    ) -> None:
        """Copy sampling_params into payload using _get_param_mapping keys."""
        for local_key, api_key in self._get_param_mapping().items():
            if local_key in sampling_params:
                payload[api_key] = sampling_params[local_key]

    def _apply_default_kwargs(self, payload: Dict[str, Any]) -> None:
        """Add default_kwargs entries that aren't already in the payload."""
        for key, value in self.default_kwargs.items():
            if key not in payload:
                payload[key] = value

"""
DeepSeek API inference provider.

Drop-in replacement for OpenRouterInference. Exposes chat()
and chat_json() with the same provider-facing interface used by the agents.

Endpoint: POST https://api.deepseek.com/v1/chat/completions
Auth: Bearer <DEEPSEEK_API_KEY>
"""

import os
import time
import random
from typing import Dict, Any, List, Optional, Tuple

try:
    import requests
except ImportError:
    raise ImportError(
        "requests module not found. Install with: pip install requests"
    )


class DeepSeekInference:
    """
    DeepSeek API inference provider.

    Environment variable required:
        DEEPSEEK_API_KEY: Your DeepSeek API key

    Usage:
        inference = DeepSeekInference("deepseek-chat")
        response = inference.chat(messages, {"temperature": 0.7, "max_tokens": 2048})
    """

    API_URL = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self,
                 model: str,
                 api_key: str = None,
                 max_retries: int = 3,
                 base_delay: float = 10.0,
                 max_delay: float = 60.0,
                 **kwargs):
        """
        Initialize DeepSeek inference provider.

        Args:
            model: Model identifier (e.g., "deepseek-chat", "deepseek-reasoner")
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            max_retries: Maximum retry attempts on transient errors (default 3)
            base_delay: Base delay in seconds for exponential backoff (default 10.0)
            max_delay: Maximum delay cap in seconds (default 60.0)
            **kwargs: Additional default parameters for requests
        """
        self.model = model
        self.model_name = model  # For compatibility with provider interface
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

        if not self.api_key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY or pass api_key."
            )

        self.max_retries = min(3, max(0, int(max_retries)))
        self.base_delay = base_delay
        self.max_delay = max_delay
        kwargs.pop("enable_caching", None)
        self.default_kwargs = kwargs

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self._session: Optional[requests.Session] = None

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=20,
                max_retries=0,
            )
            self._session.mount('https://', adapter)
        return self._session

    @staticmethod
    def _sanitize_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        DeepSeek strictly requires every assistant message with tool_calls to be
        immediately followed by tool messages for every tool_call_id.

        Strip all tool_calls from assistant messages and convert tool-role messages
        into user-role messages (preserving their content), so the model keeps full
        context without triggering DeepSeek's strict tool-call pairing check.
        """
        cleaned: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "")
            if role == "assistant" and msg.get("tool_calls"):
                stripped = {k: v for k, v in msg.items() if k != "tool_calls"}
                cleaned.append(stripped)
            elif role == "tool":
                content = msg.get("content", "")
                tc_id = msg.get("tool_call_id", "")
                tc_name = msg.get("name", "tool")
                prefix = f"[tool_result id={tc_id} name={tc_name}]\n"
                cleaned.append({"role": "user", "content": prefix + str(content)})
            else:
                cleaned.append(msg)
        return cleaned

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        sampling_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._sanitize_tool_messages(messages),
            "stream": False,
        }

        param_mapping = {
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
        for local_key, api_key in param_mapping.items():
            if local_key in sampling_params:
                payload[api_key] = sampling_params[local_key]

        for key, value in self.default_kwargs.items():
            if key not in payload:
                payload[key] = value

        return payload

    def chat(self, messages: List[Dict[str, Any]], sampling_params: Dict[str, Any]) -> Tuple[str, Dict]:
        data = self.chat_json(messages, sampling_params)
        return self._extract_chat_text_and_usage(data)

    def chat_json(self, messages: List[Dict[str, Any]], sampling_params: Dict[str, Any]) -> Dict[str, Any]:
        payload = self._build_payload(messages, sampling_params)
        return self._request_json_with_retry(payload)

    @staticmethod
    def _extract_chat_text_and_usage(data: Dict[str, Any]) -> Tuple[str, Dict]:
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

    def _request_json_with_retry(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make API request with exponential backoff retry logic.

        Retries on:
            - HTTP 429 (rate limit)
            - HTTP 500, 502, 503, 504 (server errors)
            - Connection errors
        """
        RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
        last_error: Optional[BaseException] = None
        session = self._get_session()

        for attempt in range(self.max_retries + 1):
            try:
                response = session.post(
                    self.API_URL,
                    headers=self.headers,
                    json=payload,
                    timeout=120,
                )

                if response.status_code == 200:
                    try:
                        data = response.json()
                    except Exception as e:
                        last_error = e
                        if attempt < self.max_retries:
                            delay = self._calculate_backoff(attempt)
                            print(f"  [DeepSeek] Malformed JSON response, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                            time.sleep(delay)
                            continue
                        print(f"  [DeepSeek] Malformed JSON after {self.max_retries} retries. Returning empty output.")
                        return {}

                    if "error" in data and "choices" not in data:
                        error_detail = data.get("error", data)
                        raise RuntimeError(f"DeepSeek returned error: {error_detail}")

                    if "choices" not in data:
                        error_detail = data.get("error", data)
                        raise RuntimeError(f"DeepSeek returned invalid response: {error_detail}")

                    return data

                if response.status_code in RETRYABLE_STATUS_CODES:
                    error_msg = f"HTTP {response.status_code}"
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = f"{error_msg}: {error_data['error']}"
                    except Exception:
                        pass

                    last_error = Exception(error_msg)

                    if attempt < self.max_retries:
                        delay = self._calculate_backoff(attempt)
                        print(f"  [DeepSeek] {error_msg}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                        time.sleep(delay)
                        continue

                if response.status_code in (401, 403):
                    raise ValueError(f"DeepSeek auth error (HTTP {response.status_code}). Check DEEPSEEK_API_KEY / permissions.")
                if response.status_code == 402:
                    raise ValueError(f"DeepSeek billing error (HTTP 402). Check account balance.")

                try:
                    body = response.text[:500]
                except Exception:
                    body = "<unreadable body>"
                raise RuntimeError(f"DeepSeek HTTP {response.status_code}: {body}")

            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = self._calculate_backoff(attempt)
                    print(f"  [DeepSeek] {type(e).__name__}, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})")
                    time.sleep(delay)
                    continue

        print(f"  [DeepSeek] Request failed after {self.max_retries} retries: {last_error}. Returning empty output.")
        return {}

    def _calculate_backoff(self, attempt: int) -> float:
        delay = min(self.max_delay, self.base_delay * (2 ** attempt))
        jitter = random.uniform(0.75, 1.25)
        return delay * jitter

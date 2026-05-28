"""
DeepSeek API inference provider.

Drop-in replacement for OpenRouterInference. Exposes chat()
and chat_json() with the same provider-facing interface used by the agents.

Endpoint: POST https://api.deepseek.com/v1/chat/completions
Auth: Bearer <DEEPSEEK_API_KEY>
"""

import time
from typing import Dict, Any, List, Optional

import requests

from inference.base import BaseInference


class DeepSeekInference(BaseInference):
    """
    DeepSeek API inference provider.

    Environment variable required:
        DEEPSEEK_API_KEY: Your DeepSeek API key

    Usage:
        inference = DeepSeekInference("deepseek-chat")
        response = inference.chat(messages, {"temperature": 0.7, "max_tokens": 2048})
    """

    API_URL = "https://api.deepseek.com/v1/chat/completions"
    API_KEY_ENV = "DEEPSEEK_API_KEY"

    def _post_init(self) -> None:
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

        self._apply_param_mapping(payload, sampling_params)
        self._apply_default_kwargs(payload)

        return payload

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

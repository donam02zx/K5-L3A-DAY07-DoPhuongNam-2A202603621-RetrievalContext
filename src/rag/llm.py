"""Gemini chat client with function calling, retry on transient errors and token accounting."""
from __future__ import annotations

import time
from typing import Any

from . import config as C

TRANSIENT = ("429", "500", "502", "503", "504", "RESOURCE_EXHAUSTED", "UNAVAILABLE", "DEADLINE", "overloaded")


class GeminiClient:
    def __init__(self, model: str = C.MODEL, api_key: str = C.GEMINI_API_KEY,
                 min_interval: float = C.MIN_SECONDS_BETWEEN_CALLS, max_retries: int = 6) -> None:
        if C.PROVIDER != "gemini":
            raise ValueError(f"only PROVIDER=gemini is implemented, got {C.PROVIDER}")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is empty: fill it in .env")
        from google import genai
        from google.genai import types
        self.types = types
        self.client = genai.Client(api_key=api_key)
        self.model, self.min_interval, self.max_retries = model, min_interval, max_retries
        self._last = 0.0

    def tools(self, declarations: list[dict]) -> list:
        t = self.types
        return [t.Tool(function_declarations=[
            t.FunctionDeclaration(name=d["name"], description=d["description"],
                                  parameters_json_schema=d["parameters"]) for d in declarations])]

    def call(self, contents: list, *, system: str, tools: list, allowed: list[str] | None = None) -> dict[str, Any]:
        """One model turn forced to call exactly one of `allowed` tools (mode ANY).

        Returns {'content': model Content (kept for history), 'calls': [(name, args)], 'usage': {...}}.
        Raises after retries are exhausted; never fabricates a response."""
        t = self.types
        cfg = t.GenerateContentConfig(
            system_instruction=system, tools=tools, temperature=0,
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
            tool_config=t.ToolConfig(function_calling_config=t.FunctionCallingConfig(
                mode="ANY", allowed_function_names=allowed)))
        delay = 5.0
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            started = time.perf_counter()
            try:
                resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
            except Exception as e:  # noqa: BLE001 - classify then re-raise
                if attempt < self.max_retries and any(s in str(e) for s in TRANSIENT):
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise
            u = resp.usage_metadata
            usage = {"prompt_tokens": (u.prompt_token_count or 0) if u else 0,
                     "output_tokens": (u.candidates_token_count or 0) if u else 0,
                     "thought_tokens": (getattr(u, "thoughts_token_count", 0) or 0) if u else 0,
                     "latency_s": round(time.perf_counter() - started, 2), "attempts": attempt}
            calls = [(fc.name, dict(fc.args or {})) for fc in (resp.function_calls or [])]
            content = resp.candidates[0].content if resp.candidates else None
            if not calls or content is None:
                if attempt < self.max_retries:  # empty/blocked turn: retry once more
                    continue
                raise RuntimeError(f"model returned no function call: {resp.text if resp else None}")
            return {"content": content, "calls": calls, "usage": usage}
        raise RuntimeError("unreachable")

    def tool_response(self, name: str, response: dict) -> Any:
        t = self.types
        return t.Content(role="user", parts=[t.Part.from_function_response(name=name, response=response)])

    def tool_response_with_image(self, name: str, response: dict, png: bytes) -> Any:
        t = self.types
        return t.Content(role="user", parts=[t.Part.from_function_response(name=name, response=response),
                                             t.Part.from_bytes(data=png, mime_type="image/png")])

    def user(self, text: str) -> Any:
        t = self.types
        return t.Content(role="user", parts=[t.Part(text=text)])


def cost_usd(usage: dict) -> float:
    out = usage.get("output_tokens", 0) + usage.get("thought_tokens", 0)
    return usage.get("prompt_tokens", 0) / 1e6 * C.PRICE_IN_PER_M + out / 1e6 * C.PRICE_OUT_PER_M

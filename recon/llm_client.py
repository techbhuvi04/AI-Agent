"""Shared Groq client helper.

A thin wrapper so callers keep the same shape they used with the old
Gemini client: `get_client()` returns None when no key is configured
(the graceful-degradation path both T3 and the Q&A agent rely on), and
`call(client, prompt)` returns the response text or None on failure.

The key is read from the GROQ_API_KEY environment variable only — never
hardcoded, never logged, never written to a file.
"""

import os

LLM_MODEL = os.environ.get("LLM_MODEL", "qwen/qwen3.8-27b")


class _GroqTextClient:
    """Wraps a Groq chat client so callers can do `client.generate_content(prompt).text`,
    matching the shape the old google-generativeai client had."""

    def __init__(self, client, model):
        self._client = client
        self._model = model

    def generate_content(self, prompt, json_mode=False, max_tokens=None):
        kwargs = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = self._client.chat.completions.create(**kwargs)
        return _Response(response.choices[0].message.content)


class _Response:
    def __init__(self, text):
        self.text = text


def is_available():
    return bool(os.environ.get("GROQ_API_KEY"))


def get_client(model=None):
    """Returns a client with a `.generate_content(prompt).text` interface,
    or None if GROQ_API_KEY is unset or the SDK isn't installed."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from groq import Groq

        raw_client = Groq(api_key=api_key)
        return _GroqTextClient(raw_client, model or LLM_MODEL)
    except ImportError:
        return None


def call(client, prompt):
    try:
        response = client.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return None

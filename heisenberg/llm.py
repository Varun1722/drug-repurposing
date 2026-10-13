"""The one OpenAI chat-completions client for the Python pipeline.

Three layers, each building on the one above:

  query_openai / query_openai_structured     raw (system, user) -> reply
  query_prompt / query_prompt_structured     render a prompts/*.prompt file, then call

Prefer the ``*_prompt`` helpers: prompt text lives in prompts/, not in code.
The TypeScript twin is lib/openai.ts.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Type

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from llm_schemas import to_strict_json_schema
from prompt_loader import PromptError, render_prompt
from resilience import (
    OPENAI_POLICY,
    ExternalCallError,
    compact_error,
    request_with_retry,
)

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")
OPENAI_TIMEOUT_SECONDS = 180


def _post_chat_completion(system_prompt: str, user_prompt: str, **extra) -> dict:
    """POST one chat completion and return the first choice
    (``{"message": {...}, "finish_reason": ...}``).

    Transient failures (429/5xx, timeouts, dropped connections) are retried
    with backoff; a failure that survives the retries raises
    ``ExternalCallError(source="openai")``.  Retrying here is safe because a
    chat completion has no side effects."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY environment variable is not set.",
              file=sys.stderr)
        sys.exit(1)

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # No custom temperature — some models (e.g. gpt-5.6-terra) only
        # support the default (1) and reject any other value.
        "max_completion_tokens": 8000,
        **extra,
    }

    print("[OpenAI] Sending request …")
    resp = request_with_retry("post", OPENAI_URL, source="openai",
                              json=payload, headers=headers,
                              timeout=OPENAI_TIMEOUT_SECONDS,
                              policy=OPENAI_POLICY)
    try:
        return resp.json()["choices"][0]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise ExternalCallError(
            "openai", "bad_response",
            "openai: response had no choices[0]",
        ) from exc


def query_openai(system_prompt: str, user_prompt: str) -> str:
    """Send a chat-completion request to the OpenAI API and return the
    assistant's reply text."""
    choice = _post_chat_completion(system_prompt, user_prompt)
    reply = choice["message"].get("content") or ""
    print(f"[OpenAI] Received {len(reply)} chars.")
    return reply


class LLMOutputError(Exception):
    """The API call succeeded but the reply is not a usable structured result.

    Strict mode guarantees schema-valid JSON for a normal completion, so this
    only happens when the model refused, ran out of tokens, was
    content-filtered, or returned values that fail our own pydantic
    validation.  Callers choose the fallback; ``kind`` is one of ``refusal``,
    ``truncated``, ``content_filter``, ``empty``, ``invalid_json`` or
    ``schema_mismatch``.  ``attempts`` is how many replies were tried.
    """

    def __init__(self, kind: str, message: str, attempts: int = 1):
        super().__init__(message)
        self.kind = kind
        self.attempts = attempts


# Kinds where asking again can help: the reply was empty or malformed, which a
# fresh sample usually fixes.  A refusal, a content-filter stop or a truncated
# reply would just come back the same way.
_REPAIRABLE = frozenset({"empty", "invalid_json", "schema_mismatch"})
MAX_OUTPUT_ATTEMPTS = 2


def query_openai_structured(system_prompt: str, user_prompt: str,
                            model_cls: Type[BaseModel]) -> BaseModel:
    """Chat completion constrained to ``model_cls`` via
    ``response_format: json_schema`` (strict).  Returns a validated
    ``model_cls`` instance.

    An empty or malformed reply is retried once, with a one-line description of
    what was wrong appended to the prompt so the model can correct it.

    Raises ``LLMOutputError`` for a refusal, a truncated/filtered reply, or a
    reply that is still unusable after the second try; ``ExternalCallError``
    when the API itself could not be reached (after retries).
    """
    prompt = user_prompt
    for attempt in range(1, MAX_OUTPUT_ATTEMPTS + 1):
        try:
            return _structured_once(system_prompt, prompt, model_cls)
        except LLMOutputError as e:
            e.attempts = attempt
            if e.kind not in _REPAIRABLE or attempt == MAX_OUTPUT_ATTEMPTS:
                raise
            print(f"[OpenAI] Reply unusable ({e.kind}); asking again.",
                  file=sys.stderr)
            prompt = (
                f"{user_prompt}\n\nYour previous reply was rejected: "
                f"{compact_error(str(e), 300)}\nReply again with JSON that satisfies "
                f"the schema."
            )
    raise AssertionError("unreachable")  # the loop always returns or raises


def _structured_once(system_prompt: str, user_prompt: str,
                     model_cls: Type[BaseModel]) -> BaseModel:
    choice = _post_chat_completion(
        system_prompt,
        user_prompt,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": model_cls.__name__,
                "strict": True,
                "schema": to_strict_json_schema(model_cls),
            },
        },
    )
    message = choice.get("message") or {}
    finish_reason = choice.get("finish_reason")

    if message.get("refusal"):
        raise LLMOutputError("refusal", f"Model refused: {message['refusal']}")
    if finish_reason == "length":
        raise LLMOutputError(
            "truncated",
            "Reply hit max_completion_tokens before the JSON was complete",
        )
    if finish_reason == "content_filter":
        raise LLMOutputError("content_filter", "Reply stopped by the content filter")
    content = message.get("content")
    if not content:
        raise LLMOutputError("empty", "Model returned no content")

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise LLMOutputError("invalid_json", f"Reply was not valid JSON: {e}") from e
    try:
        result = model_cls.model_validate(data)
    except ValidationError as e:
        raise LLMOutputError("schema_mismatch", f"Reply failed validation: {e}") from e
    print(f"[OpenAI] Received structured {model_cls.__name__}.")
    return result


def query_prompt(prompt_id: str, **variables) -> str:
    """Render ``prompts/<prompt_id>.prompt`` and return the model's free-text reply."""
    prompt = render_prompt(prompt_id, **variables)
    if prompt.output_schema is not None:
        raise PromptError(
            f"{prompt_id} declares output.schema {prompt.output_schema!r}; "
            f"call query_prompt_structured instead"
        )
    return query_openai(prompt.system, prompt.user)


def query_prompt_structured(prompt_id: str, model_cls: Type[BaseModel],
                            **variables) -> BaseModel:
    """Render ``prompts/<prompt_id>.prompt`` and return the reply as ``model_cls``.

    The prompt file must declare ``output.schema: <model_cls.__name__>``, which
    keeps each prompt tied to the schema it was written for.
    """
    prompt = render_prompt(prompt_id, **variables)
    if prompt.output_schema != model_cls.__name__:
        raise PromptError(
            f"{prompt_id} declares output.schema {prompt.output_schema!r} "
            f"but was called with {model_cls.__name__}"
        )
    return query_openai_structured(prompt.system, prompt.user, model_cls)

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

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from llm_schemas import to_strict_json_schema
from prompt_loader import PromptError, render_prompt

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")
OPENAI_TIMEOUT_SECONDS = 180


def _post_chat_completion(system_prompt: str, user_prompt: str, **extra) -> dict:
    """POST one chat completion and return the first choice
    (``{"message": {...}, "finish_reason": ...}``)."""
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
    resp = requests.post(OPENAI_URL, json=payload, headers=headers,
                         timeout=OPENAI_TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()["choices"][0]


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
    ``schema_mismatch``.
    """

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def query_openai_structured(system_prompt: str, user_prompt: str,
                            model_cls: Type[BaseModel]) -> BaseModel:
    """Chat completion constrained to ``model_cls`` via
    ``response_format: json_schema`` (strict).  Returns a validated
    ``model_cls`` instance.

    Raises ``LLMOutputError`` for a refusal, truncated/filtered/empty reply, or
    a reply that fails validation.  HTTP/network failures raise the usual
    ``requests`` exceptions.
    """
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

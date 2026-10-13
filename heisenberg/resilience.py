"""Retry, failure classification and error recording for every external call.

Every network dependency of the pipeline (OpenAI, PubChem, RCSB, PubMed, Exa,
the GNINA worker) goes through here, so they all fail the same way:

  * Transient failures (timeouts, connection errors, HTTP 408/409/429/5xx) are
    retried with capped exponential backoff and full jitter, honouring a
    ``Retry-After`` header.  Attempts *and* total time are bounded.
  * Failures that waiting cannot fix (400/401/403/404, and 429s that mean the
    credit balance or a spend limit is exhausted) are raised at once.
  * When the call still fails, ``ExternalCallError`` is raised.  "Nothing
    found" (a 404 the caller allows) is *not* an error: it is returned as a
    normal response, so callers can tell "empty" from "failed".

``record_error`` / ``FailureGuard`` then turn those failures into entries in
``pipeline_state.json`` and into a stop-and-ask-a-human after repeated
failures.  The TypeScript twin is lib/retry.ts.
"""

from __future__ import annotations

import email.utils
import http.client
import json
import random
import re
import socket
import sys
import time
import urllib.error
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable, Optional, TypeVar

import requests

T = TypeVar("T")

# A 429 whose body carries one of these codes means "no credit / over a spend
# limit".  Retrying cannot restore access (OpenAI error-codes guide), so it is
# treated as permanent.
BILLING_CODES = frozenset({
    "insufficient_quota",
    "credit_balance_exhausted",
    "organization_usage_limit_exceeded",
    "organization_spend_limit_exceeded",
    "project_spend_limit_exceeded",
})


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to try before giving up on one call."""

    max_attempts: int = 4           # first try + 3 retries
    base_delay: float = 1.0         # seconds; doubles per retry, then jittered
    max_delay: float = 30.0         # cap on one wait
    max_total_seconds: float = 120.0  # cap on attempts + waits together
    max_retry_after: float = 60.0   # a longer Retry-After is surfaced, not slept


# Public data APIs (PubChem, RCSB, PubMed, Exa).
DEFAULT_POLICY = RetryPolicy()
# Chat completions can legitimately run for minutes; the budget covers that.
OPENAI_POLICY = RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=30.0,
                            max_total_seconds=600.0)
# One docking chunk is minutes of compute, so retry fewer times but wait longer.
DOCKING_POLICY = RetryPolicy(max_attempts=3, base_delay=5.0, max_delay=60.0,
                             max_total_seconds=1500.0)


class ExternalCallError(Exception):
    """An external call failed for good: out of retries, or not worth retrying.

    ``source`` names the dependency (``openai``, ``pubchem``, ``rcsb``,
    ``pubmed``, ``exa``, ``gnina``).  ``kind`` is one of ``timeout``,
    ``connection``, ``rate_limited``, ``quota``, ``auth``, ``server``,
    ``conflict``, ``client``, ``bad_response``, ``not_found``.
    """

    def __init__(self, source: str, kind: str, message: str, *,
                 status: Optional[int] = None, attempts: int = 1,
                 retryable: bool = False):
        super().__init__(message)
        self.source = source
        self.kind = kind
        self.status = status
        self.attempts = attempts
        self.retryable = retryable


class EscalationRequired(Exception):
    """Too many consecutive failures: stop and hand the decision to a human."""

    def __init__(self, source: str, count: int, last_error: Exception):
        self.source = source
        self.count = count
        self.last_error = last_error
        super().__init__(
            f"{count} consecutive failures calling {source} "
            f"(last: {compact_error(last_error)}). Progress is saved; fix the "
            f"cause and re-run with --resume."
        )


class _HttpStatus(Exception):
    """Internal: a response whose status says the call failed."""

    def __init__(self, status: int, body: str = "",
                 retry_after: Optional[float] = None):
        super().__init__(f"HTTP {status}")
        self.status = status
        self.body = body
        self.retry_after = retry_after


@dataclass(frozen=True)
class _Failure:
    kind: str
    retryable: bool
    status: Optional[int]
    retry_after: Optional[float]
    detail: str


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

# Anything key-shaped is scrubbed from error text: error bodies can echo a
# (masked) API key, and these messages end up in pipeline_state.json.
_SECRET = re.compile(
    r"\b(?:sk|pk|rk)-[A-Za-z0-9_*.\-]{3,}"
    r"|(?:api[_-]?key|token|bearer)[=:\s]+[A-Za-z0-9_*.\-]{6,}",
    re.IGNORECASE,
)


def _scrub(text: str) -> str:
    return _SECRET.sub("[redacted]", text)


def _snippet(body: str, limit: int = 160) -> str:
    """The service's own explanation from an error body (OpenAI
    ``error.message``, PubChem ``Fault.Message``, or short plain text), one
    line.  Empty if there is none; HTML error pages are not quoted."""
    if not body or body.lstrip().startswith("<"):
        return ""
    try:
        data = json.loads(body)
    except ValueError:
        text = body
    else:
        err = data.get("error") if isinstance(data, dict) else None
        fault = data.get("Fault") if isinstance(data, dict) else None
        message = err.get("message") if isinstance(err, dict) else err
        if not isinstance(message, str) and isinstance(fault, dict):
            message = fault.get("Message")
        text = message if isinstance(message, str) else ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"

def _is_billing(body: str) -> bool:
    """True if a 429 body says credit/spend/quota is exhausted (not a rate limit)."""
    try:
        err = json.loads(body).get("error") or {}
        codes = {err.get("code"), err.get("type")}
        return bool(codes & BILLING_CODES)
    except (ValueError, AttributeError, TypeError):
        return any(code in body for code in BILLING_CODES)


def _classify_status(status: int, body: str = "") -> tuple[str, bool]:
    """(kind, retryable) for an HTTP error status."""
    if status == 429:
        return ("quota", False) if _is_billing(body) else ("rate_limited", True)
    if status == 408:
        return "timeout", True
    if status == 409:
        return "conflict", True
    if status >= 500:
        return "server", True
    if status in (401, 403):
        return "auth", False
    return "client", False


def _parse_retry_after(headers) -> Optional[float]:
    """Seconds to wait from ``retry-after-ms`` / ``retry-after`` (seconds or
    an HTTP date), or None."""
    if headers is None:
        return None

    def header(name: str):
        try:
            value = headers.get(name)
        except Exception:
            return None
        return value if isinstance(value, (str, int, float)) else None

    ms = header("retry-after-ms")
    if ms is not None:
        try:
            return max(0.0, float(ms) / 1000.0)
        except ValueError:
            pass
    value = header("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    parsed = email.utils.parsedate_tz(str(value))
    if parsed is None:
        return None
    return max(0.0, email.utils.mktime_tz(parsed) - time.time())


def _http_detail(status: int, body: str) -> str:
    snippet = _snippet(body)
    return f"HTTP {status}: {snippet}" if snippet else f"HTTP {status}"


def _classify(exc: Exception) -> Optional[_Failure]:
    """Classify an exception from one attempt, or None if it is not a
    transport failure (a bug, say) and must propagate untouched."""
    if isinstance(exc, _HttpStatus):
        kind, retryable = _classify_status(exc.status, exc.body)
        return _Failure(kind, retryable, exc.status, exc.retry_after,
                        _http_detail(exc.status, exc.body))

    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace")[:2000]
        except Exception:
            body = ""
        kind, retryable = _classify_status(exc.code, body)
        return _Failure(kind, retryable, exc.code,
                        _parse_retry_after(exc.headers), _http_detail(exc.code, body))

    if isinstance(exc, requests.HTTPError) and getattr(exc, "response", None) is not None:
        status = exc.response.status_code
        kind, retryable = _classify_status(status)
        return _Failure(kind, retryable, status, None, f"HTTP {status}")

    if isinstance(exc, urllib.error.URLError):
        if isinstance(exc.reason, (socket.timeout, TimeoutError)):
            return _Failure("timeout", True, None, None, "timed out")
        return _Failure("connection", True, None, None, str(exc.reason))

    # requests.ConnectTimeout is both a Timeout and a ConnectionError: Timeout first.
    if isinstance(exc, (requests.Timeout, socket.timeout, TimeoutError)):
        return _Failure("timeout", True, None, None, "timed out")
    if isinstance(exc, (requests.ConnectionError, ConnectionError,
                        requests.exceptions.ChunkedEncodingError,
                        http.client.HTTPException, socket.gaierror)):
        return _Failure("connection", True, None, None, compact_error(exc, 120))

    if isinstance(exc, requests.RequestException):
        # Bad URL, too many redirects, ...: a request we built wrong; waiting won't help.
        return _Failure("client", False, None, None, compact_error(exc, 120))
    return None


# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------

def _backoff_delay(attempt: int, policy: RetryPolicy,
                   retry_after: Optional[float],
                   rng: Callable[[], float]) -> Optional[float]:
    """Seconds to wait before the next try, or None to give up now.

    Full jitter: a uniform draw from [0, min(max_delay, base * 2**(attempt-1))]
    spreads out clients that failed together.  A server ``Retry-After`` is a
    floor, and one longer than ``max_retry_after`` is not slept on.
    """
    cap = min(policy.max_delay, policy.base_delay * (2 ** (attempt - 1)))
    delay = rng() * cap
    if retry_after is not None:
        if retry_after > policy.max_retry_after:
            return None
        delay = max(delay, retry_after)
    return delay


def call_with_retry(fn: Callable[[], T], *, source: str,
                    policy: RetryPolicy = DEFAULT_POLICY, label: str = "",
                    rng: Optional[Callable[[], float]] = None) -> T:
    """Run ``fn`` until it succeeds, retrying transient failures per ``policy``.

    Raises ``ExternalCallError`` when the failure is permanent or the attempt /
    time budget runs out.  Exceptions that are not transport failures
    propagate unchanged and are never retried.  ``fn`` must be safe to repeat.
    """
    rng = rng or random.random
    what = f"{source} {label}".strip()
    started = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except Exception as exc:
            failure = _classify(exc)
            if failure is None:
                raise
            error = ExternalCallError(
                source, failure.kind,
                _scrub(f"{what}: {failure.kind} ({failure.detail}) after {attempt} attempt(s)"),
                status=failure.status, attempts=attempt, retryable=failure.retryable,
            )
            if not failure.retryable or attempt >= policy.max_attempts:
                raise error from exc
            delay = _backoff_delay(attempt, policy, failure.retry_after, rng)
            if delay is None or (time.monotonic() - started) + delay > policy.max_total_seconds:
                raise error from exc
            print(_scrub(f"[retry] {what}: {failure.kind} ({failure.detail}); "
                         f"attempt {attempt}/{policy.max_attempts}, retrying in {delay:.1f}s"),
                  flush=True)
            time.sleep(delay)


def _body_text(resp) -> str:
    text = getattr(resp, "text", "")
    return text[:2000] if isinstance(text, str) else ""


def request_with_retry(method: str, url: str, *, source: str,
                       allow_status: Iterable[int] = (),
                       policy: RetryPolicy = DEFAULT_POLICY,
                       **kwargs) -> requests.Response:
    """``requests.<method>(url, **kwargs)`` with retries.

    Returns the response for any status below 400 and for statuses listed in
    ``allow_status`` (e.g. 404 "not found" or 204 "no content": results, not
    failures).  Any other status counts as a failed attempt.
    """
    allowed = frozenset(allow_status)

    def attempt() -> requests.Response:
        resp = getattr(requests, method.lower())(url, **kwargs)
        status = resp.status_code
        if status < 400 or status in allowed:
            return resp
        raise _HttpStatus(status, _body_text(resp), _parse_retry_after(getattr(resp, "headers", None)))

    # Strip the query string: it can carry API keys.
    return call_with_retry(attempt, source=source, policy=policy,
                           label=f"{method.upper()} {url.split('?')[0]}")


def response_json(resp, source: str):
    """``resp.json()``, raising ExternalCallError instead of ValueError for a
    body that is not JSON."""
    try:
        return resp.json()
    except ValueError as exc:
        raise ExternalCallError(source, "bad_response",
                                f"{source}: response was not valid JSON") from exc


def run_cli(main: Callable[[], None]) -> None:
    """Run a script's ``main()``; an ``ExternalCallError`` becomes one error
    line and exit code 1 instead of a traceback."""
    try:
        main()
    except ExternalCallError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Recording errors in pipeline state
# ---------------------------------------------------------------------------

def compact_error(error, limit: int = 240) -> str:
    """One short line for an error: class name (unless ours) and message,
    whitespace collapsed, truncated.  This is what goes into state and into
    LLM context, never a stack trace."""
    text = " ".join(str(error).split())
    if isinstance(error, BaseException) and not isinstance(
            error, (ExternalCallError, EscalationRequired)):
        text = f"{type(error).__name__}: {text}"
    text = _scrub(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def record_error(state: dict, *, stage: str, source: str, kind: str,
                 message: str, target: Optional[str] = None,
                 fatal: bool = False, **extra) -> dict:
    """Append a compact error entry to ``state["errors"]`` and return it."""
    entry = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "source": source,
        "kind": kind,
        "message": compact_error(message),
        "fatal": fatal,
    }
    if target is not None:
        entry["target"] = target
    entry.update(extra)
    state.setdefault("errors", []).append(entry)
    return entry


def record_exception(state: dict, *, stage: str, exc: Exception,
                     target: Optional[str] = None, fatal: bool = False) -> dict:
    """``record_error`` for a caught exception, using its source/kind when it
    is an ExternalCallError."""
    if isinstance(exc, ExternalCallError):
        extra = {"attempts": exc.attempts}
        if exc.status is not None:
            extra["status"] = exc.status
        return record_error(state, stage=stage, source=exc.source, kind=exc.kind,
                            message=str(exc), target=target, fatal=fatal, **extra)
    source = getattr(exc, "source", "pipeline")
    kind = getattr(exc, "kind", type(exc).__name__)
    return record_error(state, stage=stage, source=source, kind=kind,
                        message=compact_error(exc), target=target, fatal=fatal)


def summarize_errors(errors: list[dict], limit: int = 8) -> list[str]:
    """Group non-fatal errors by (stage, source, kind) into short lines for a
    report: ``docking / gnina / server x2 (KRAS, EGFR)``."""
    groups: dict[tuple, list[dict]] = {}
    for e in errors:
        if e.get("fatal"):
            continue
        groups.setdefault((e["stage"], e["source"], e["kind"]), []).append(e)
    lines = []
    for (stage, source, kind), items in groups.items():
        targets = list(dict.fromkeys(i["target"] for i in items if i.get("target")))
        suffix = f" ({', '.join(targets[:4])}{', …' if len(targets) > 4 else ''})" if targets else ""
        count = f" x{len(items)}" if len(items) > 1 else ""
        lines.append(f"{stage} / {source} / {kind}{count}{suffix}: {items[0]['message']}")
    return lines[:limit]


class FailureGuard:
    """Count consecutive failures to one source and escalate at a threshold.

    Call ``ok()`` after each success and ``failed(exc)`` after each failure;
    ``failed`` raises ``EscalationRequired`` once ``threshold`` failures happen
    in a row.  A source that keeps failing after full retries will not recover
    by trying the next item, so the run stops with its state saved rather than
    grinding through the rest.
    """

    def __init__(self, source: str, threshold: int = 3):
        self.source = source
        self.threshold = threshold
        self.consecutive = 0

    def ok(self) -> None:
        self.consecutive = 0

    def failed(self, exc: Exception) -> None:
        self.consecutive += 1
        if self.consecutive >= self.threshold:
            raise EscalationRequired(self.source, self.consecutive, exc)

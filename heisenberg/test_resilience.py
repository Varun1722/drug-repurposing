"""Tests for resilience.py: retry/backoff, failure classification, and the
helpers that turn failures into pipeline-state entries.

No network and no real sleeping: ``time.sleep`` is replaced by a recorder.
"""

import json
import socket
import urllib.error
from email.message import Message
from unittest.mock import MagicMock, patch

import pytest
import requests

import resilience
from resilience import (
    EscalationRequired,
    ExternalCallError,
    FailureGuard,
    RetryPolicy,
    call_with_retry,
    compact_error,
    record_error,
    record_exception,
    request_with_retry,
    response_json,
    run_cli,
    summarize_errors,
)


@pytest.fixture
def sleeps(monkeypatch):
    """Record backoff waits instead of sleeping."""
    waited = []
    monkeypatch.setattr("time.sleep", waited.append)
    return waited


def _flaky(failures, exc_factory, result="ok"):
    """A function that raises ``exc_factory()`` ``failures`` times, then returns."""
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc_factory()
        return result

    fn.calls = calls
    return fn


def _conn_err():
    return requests.ConnectionError("connection reset")


def _http_error(code, body=b"", headers=None):
    msg = Message()
    for key, value in (headers or {}).items():
        msg[key] = value
    import io

    return urllib.error.HTTPError("http://x", code, "err", msg, io.BytesIO(body))


def _resp(status, text="", headers=None):
    r = MagicMock()
    r.status_code = status
    r.text = text
    r.headers = headers or {}
    return r


FULL = lambda: 1.0  # noqa: E731  rng that always draws the maximum


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------

def test_success_needs_no_retry(sleeps):
    fn = _flaky(0, _conn_err)
    assert call_with_retry(fn, source="x") == "ok"
    assert fn.calls["n"] == 1 and sleeps == []


def test_retries_transient_failures_then_succeeds(sleeps):
    fn = _flaky(2, _conn_err)
    assert call_with_retry(fn, source="x", rng=FULL) == "ok"
    assert fn.calls["n"] == 3
    assert len(sleeps) == 2


def test_backoff_doubles_each_retry(sleeps):
    fn = _flaky(99, _conn_err)
    with pytest.raises(ExternalCallError):
        call_with_retry(fn, source="x", rng=FULL,
                        policy=RetryPolicy(max_attempts=4, base_delay=1.0))
    assert sleeps == [1.0, 2.0, 4.0]


def test_full_jitter_scales_the_wait_down(sleeps):
    fn = _flaky(99, _conn_err)
    with pytest.raises(ExternalCallError):
        call_with_retry(fn, source="x", rng=lambda: 0.5,
                        policy=RetryPolicy(max_attempts=4, base_delay=1.0))
    assert sleeps == [0.5, 1.0, 2.0]


def test_wait_is_capped_at_max_delay(sleeps):
    fn = _flaky(99, _conn_err)
    with pytest.raises(ExternalCallError):
        call_with_retry(fn, source="x", rng=FULL,
                        policy=RetryPolicy(max_attempts=6, base_delay=10.0, max_delay=25.0))
    assert sleeps == [10.0, 20.0, 25.0, 25.0, 25.0]


def test_default_jitter_stays_within_the_cap(sleeps):
    fn = _flaky(99, _conn_err)
    with pytest.raises(ExternalCallError):
        call_with_retry(fn, source="x",
                        policy=RetryPolicy(max_attempts=5, base_delay=1.0, max_delay=3.0))
    assert all(0 <= w <= 3.0 for w in sleeps) and len(sleeps) == 4


def test_gives_up_after_max_attempts_with_the_attempt_count(sleeps):
    fn = _flaky(99, _conn_err)
    with pytest.raises(ExternalCallError) as exc:
        call_with_retry(fn, source="pubchem", label="GET /x",
                        policy=RetryPolicy(max_attempts=3))
    e = exc.value
    assert (e.source, e.kind, e.attempts, e.retryable) == ("pubchem", "connection", 3, True)
    assert fn.calls["n"] == 3
    assert "3 attempt(s)" in str(e) and "GET /x" in str(e)
    assert isinstance(e.__cause__, requests.ConnectionError)


def test_total_time_budget_stops_retrying(monkeypatch, sleeps):
    clock = {"t": 0.0}
    monkeypatch.setattr("time.monotonic", lambda: clock["t"])
    monkeypatch.setattr("time.sleep", lambda s: clock.__setitem__("t", clock["t"] + s))

    def slow_failure():
        clock["t"] += 10.0          # each attempt burns 10s
        raise _conn_err()

    with pytest.raises(ExternalCallError) as exc:
        call_with_retry(slow_failure, source="x", rng=FULL,
                        policy=RetryPolicy(max_attempts=10, base_delay=1.0,
                                           max_total_seconds=25.0))
    # t=10 fail, wait 1 (11) -> t=21 fail, wait 2 (23) -> t=33 fail: the next
    # wait would end at 33+4 > 25, so it stops after the 3rd attempt, not the 10th.
    assert exc.value.attempts == 3


def test_non_transport_errors_propagate_untouched_and_are_not_retried(sleeps):
    fn = _flaky(5, lambda: KeyError("bug"))
    with pytest.raises(KeyError):
        call_with_retry(fn, source="x")
    assert fn.calls["n"] == 1 and sleeps == []


def test_keyboard_interrupt_is_never_swallowed(sleeps):
    def fn():
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        call_with_retry(fn, source="x")


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,kind,retryable", [
    (400, "client", False),
    (401, "auth", False),
    (403, "auth", False),
    (404, "client", False),
    (422, "client", False),
    (408, "timeout", True),
    (409, "conflict", True),
    (429, "rate_limited", True),
    (500, "server", True),
    (502, "server", True),
    (503, "server", True),
    (504, "server", True),
])
def test_http_status_classification(status, kind, retryable, sleeps):
    with patch("requests.get", return_value=_resp(status)) as get:
        with pytest.raises(ExternalCallError) as exc:
            request_with_retry("get", "http://x/y", source="svc",
                               policy=RetryPolicy(max_attempts=3))
    assert (exc.value.kind, exc.value.retryable, exc.value.status) == (kind, retryable, status)
    assert get.call_count == (3 if retryable else 1)


@pytest.mark.parametrize("code", sorted(resilience.BILLING_CODES))
def test_429_with_a_billing_code_is_permanent(code, sleeps):
    body = json.dumps({"error": {"code": code, "type": "insufficient_quota"}})
    with patch("requests.post", return_value=_resp(429, body)) as post:
        with pytest.raises(ExternalCallError) as exc:
            request_with_retry("post", "http://x", source="openai")
    assert exc.value.kind == "quota" and not exc.value.retryable
    assert post.call_count == 1 and sleeps == []


def test_429_billing_code_in_a_non_json_body_is_still_recognised(sleeps):
    with patch("requests.post", return_value=_resp(429, "oops insufficient_quota oops")):
        with pytest.raises(ExternalCallError) as exc:
            request_with_retry("post", "http://x", source="openai")
    assert exc.value.kind == "quota"


def test_ordinary_429_is_a_rate_limit(sleeps):
    body = json.dumps({"error": {"code": "rate_limit_exceeded", "type": "requests"}})
    with patch("requests.post", return_value=_resp(429, body)):
        with pytest.raises(ExternalCallError) as exc:
            request_with_retry("post", "http://x", source="openai",
                               policy=RetryPolicy(max_attempts=2))
    assert exc.value.kind == "rate_limited" and exc.value.retryable


@pytest.mark.parametrize("exc_factory,kind", [
    (lambda: requests.ConnectTimeout("t"), "timeout"),
    (lambda: requests.ReadTimeout("t"), "timeout"),
    (lambda: socket.timeout("t"), "timeout"),
    (lambda: TimeoutError("t"), "timeout"),
    (lambda: requests.ConnectionError("c"), "connection"),
    (lambda: ConnectionResetError("c"), "connection"),
    (lambda: urllib.error.URLError("dns failure"), "connection"),
    (lambda: urllib.error.URLError(socket.timeout("t")), "timeout"),
])
def test_transport_exception_classification(exc_factory, kind, sleeps):
    fn = _flaky(99, exc_factory)
    with pytest.raises(ExternalCallError) as exc:
        call_with_retry(fn, source="x", policy=RetryPolicy(max_attempts=2))
    assert exc.value.kind == kind and exc.value.retryable


def test_malformed_request_is_not_retried(sleeps):
    fn = _flaky(99, lambda: requests.exceptions.InvalidURL("bad"))
    with pytest.raises(ExternalCallError) as exc:
        call_with_retry(fn, source="x")
    assert exc.value.kind == "client" and fn.calls["n"] == 1


def test_urllib_http_errors_are_classified_by_code(sleeps):
    fn = _flaky(99, lambda: _http_error(429))
    with pytest.raises(ExternalCallError) as exc:
        call_with_retry(fn, source="pubmed", policy=RetryPolicy(max_attempts=2))
    assert (exc.value.kind, exc.value.status, exc.value.attempts) == ("rate_limited", 429, 2)

    fn = _flaky(99, lambda: _http_error(400))
    with pytest.raises(ExternalCallError) as exc:
        call_with_retry(fn, source="pubmed")
    assert exc.value.kind == "client" and fn.calls["n"] == 1


# ---------------------------------------------------------------------------
# Retry-After
# ---------------------------------------------------------------------------

def test_retry_after_seconds_is_a_floor_for_the_wait(sleeps):
    fn = _flaky(1, lambda: _http_error(429, headers={"Retry-After": "5"}))
    assert call_with_retry(fn, source="x", rng=lambda: 0.0) == "ok"
    assert sleeps == [5.0]


def test_retry_after_longer_than_the_cap_is_not_slept_on(sleeps):
    fn = _flaky(99, lambda: _http_error(429, headers={"Retry-After": "3600"}))
    with pytest.raises(ExternalCallError) as exc:
        call_with_retry(fn, source="x", policy=RetryPolicy(max_retry_after=60))
    assert exc.value.kind == "rate_limited"
    assert fn.calls["n"] == 1 and sleeps == []


def test_retry_after_that_does_not_fit_the_time_budget_is_not_slept_on(sleeps):
    fn = _flaky(99, lambda: _http_error(429, headers={"Retry-After": "50"}))
    with pytest.raises(ExternalCallError):
        call_with_retry(fn, source="x", policy=RetryPolicy(max_total_seconds=10))
    assert sleeps == []


@pytest.mark.parametrize("headers,expected", [
    ({"retry-after-ms": "1500"}, 1.5),
    ({"retry-after-ms": "1500", "retry-after": "9"}, 1.5),   # ms is more precise
    ({"retry-after": "7"}, 7.0),
    ({"retry-after": "2.5"}, 2.5),
    ({"retry-after": "-3"}, 0.0),
    ({"retry-after": "soon"}, None),
    ({}, None),
])
def test_parse_retry_after(headers, expected):
    assert resilience._parse_retry_after(headers) == expected


def test_parse_retry_after_http_date(monkeypatch):
    monkeypatch.setattr("time.time", lambda: 1445412480.0)  # 2015-10-21 07:28:00 UTC
    value = "Wed, 21 Oct 2015 07:28:10 GMT"
    assert resilience._parse_retry_after({"retry-after": value}) == pytest.approx(10.0)
    past = "Wed, 21 Oct 2015 07:00:00 GMT"
    assert resilience._parse_retry_after({"retry-after": past}) == 0.0


def test_parse_retry_after_ignores_mock_headers():
    assert resilience._parse_retry_after(MagicMock()) is None
    assert resilience._parse_retry_after(None) is None


# ---------------------------------------------------------------------------
# request_with_retry
# ---------------------------------------------------------------------------

def test_allowed_status_is_a_result_not_a_failure(sleeps):
    with patch("requests.get", return_value=_resp(404)) as get:
        resp = request_with_retry("get", "http://x", source="pubchem", allow_status=(404,))
    assert resp.status_code == 404 and get.call_count == 1


def test_a_503_then_a_200_succeeds(sleeps):
    with patch("requests.get", side_effect=[_resp(503), _resp(200)]) as get:
        resp = request_with_retry("get", "http://x", source="pubchem")
    assert resp.status_code == 200 and get.call_count == 2 and len(sleeps) == 1


def test_request_kwargs_are_passed_through(sleeps):
    with patch("requests.post", return_value=_resp(200)) as post:
        request_with_retry("POST", "http://x", source="s", json={"a": 1},
                           headers={"h": "v"}, timeout=7)
    post.assert_called_once_with("http://x", json={"a": 1}, headers={"h": "v"}, timeout=7)


def test_secrets_in_the_query_string_never_reach_logs_or_errors(sleeps, capsys):
    url = "http://ncbi/esearch?term=cancer&api_key=SECRET123"
    with patch("requests.get", return_value=_resp(503)):
        with pytest.raises(ExternalCallError) as exc:
            request_with_retry("get", url, source="pubmed",
                               policy=RetryPolicy(max_attempts=2))
    assert "SECRET123" not in str(exc.value)
    assert "SECRET123" not in capsys.readouterr().out


def test_retry_notice_is_printed(sleeps, capsys):
    with patch("requests.get", side_effect=[_resp(503), _resp(200)]):
        request_with_retry("get", "http://x/y", source="pubchem")
    out = capsys.readouterr().out
    assert "[retry] pubchem GET http://x/y" in out and "attempt 1/4" in out


def test_response_json_wraps_invalid_json():
    bad = MagicMock()
    bad.json.side_effect = ValueError("not json")
    with pytest.raises(ExternalCallError) as exc:
        response_json(bad, "pubchem")
    assert (exc.value.source, exc.value.kind) == ("pubchem", "bad_response")


def test_run_cli_turns_an_external_failure_into_a_clean_exit(capsys):
    def main():
        raise ExternalCallError("exa", "server", "exa: server (HTTP 500) after 4 attempt(s)")

    with pytest.raises(SystemExit) as exc:
        run_cli(main)
    assert exc.value.code == 1
    assert "ERROR: exa: server" in capsys.readouterr().err


def test_run_cli_leaves_other_errors_alone():
    def main():
        raise ValueError("a bug")

    with pytest.raises(ValueError):
        run_cli(main)


# ---------------------------------------------------------------------------
# Recording errors in state
# ---------------------------------------------------------------------------

def test_compact_error_is_one_short_line():
    err = ValueError("line one\n   line two\n\tline three")
    assert compact_error(err) == "ValueError: line one line two line three"
    assert compact_error(ExternalCallError("s", "k", "msg")) == "msg"   # no class prefix
    long = compact_error("x" * 1000, limit=50)
    assert len(long) == 50 and long.endswith("…")


def test_record_error_appends_a_complete_entry():
    state = {}
    entry = record_error(state, stage="docking", source="gnina", kind="server",
                         message="boom", target="KRAS", chunk=2)
    assert state["errors"] == [entry]
    assert {"at", "stage", "source", "kind", "message", "fatal", "target", "chunk"} <= set(entry)
    assert entry["fatal"] is False and entry["chunk"] == 2


def test_record_exception_uses_the_external_error_fields():
    state = {"errors": []}
    e = ExternalCallError("pubchem", "rate_limited", "pubchem: 429", status=429, attempts=4)
    entry = record_exception(state, stage="structure", exc=e, target="KRAS/x", fatal=True)
    assert (entry["source"], entry["kind"], entry["status"], entry["attempts"]) == (
        "pubchem", "rate_limited", 429, 4)
    assert entry["fatal"] is True and entry["target"] == "KRAS/x"


def test_record_exception_handles_arbitrary_exceptions():
    state = {}
    entry = record_exception(state, stage="report", exc=RuntimeError("kaboom"))
    assert (entry["source"], entry["kind"]) == ("pipeline", "RuntimeError")
    assert entry["message"] == "RuntimeError: kaboom"


def test_summarize_errors_groups_and_skips_fatal():
    errors = [
        {"stage": "structure", "source": "pubchem", "kind": "server", "message": "m1", "target": "A/x", "fatal": False},
        {"stage": "structure", "source": "pubchem", "kind": "server", "message": "m2", "target": "A/y", "fatal": False},
        {"stage": "docking", "source": "gnina", "kind": "ligand_failed", "message": "3 ligands", "fatal": False},
        {"stage": "report", "source": "openai", "kind": "server", "message": "dead", "fatal": True},
    ]
    lines = summarize_errors(errors)
    assert lines == [
        "structure / pubchem / server x2 (A/x, A/y): m1",
        "docking / gnina / ligand_failed: 3 ligands",
    ]


def test_summarize_errors_limits_the_lines():
    errors = [{"stage": "s", "source": f"src{i}", "kind": "k", "message": "m", "fatal": False}
              for i in range(20)]
    assert len(summarize_errors(errors, limit=5)) == 5


# ---------------------------------------------------------------------------
# FailureGuard
# ---------------------------------------------------------------------------

def test_guard_escalates_on_the_third_consecutive_failure():
    guard = FailureGuard("pubchem", threshold=3)
    err = ExternalCallError("pubchem", "server", "pubchem: server")
    guard.failed(err)
    guard.failed(err)
    with pytest.raises(EscalationRequired) as exc:
        guard.failed(err)
    assert exc.value.source == "pubchem" and exc.value.count == 3
    assert "--resume" in str(exc.value) and "3 consecutive failures calling pubchem" in str(exc.value)


def test_a_success_resets_the_count():
    guard = FailureGuard("rcsb", threshold=3)
    err = ExternalCallError("rcsb", "server", "x")
    guard.failed(err)
    guard.failed(err)
    guard.ok()
    guard.failed(err)
    guard.failed(err)       # 2 in a row since the reset: no escalation
    assert guard.consecutive == 2


# ---------------------------------------------------------------------------
# Error text: the service's own explanation, and no secrets
# ---------------------------------------------------------------------------

def test_the_services_own_explanation_is_in_the_error(sleeps):
    body = json.dumps({"error": {"message": "The model `nope` does not exist", "code": "model_not_found"}})
    with patch("requests.post", return_value=_resp(404, body)):
        with pytest.raises(ExternalCallError) as exc:
            request_with_retry("post", "http://x", source="openai")
    assert "client (HTTP 404: The model `nope` does not exist)" in str(exc.value)


def test_pubchem_style_fault_messages_are_quoted(sleeps):
    body = json.dumps({"Fault": {"Code": "PUGREST.BadRequest", "Message": "Invalid name"}})
    with patch("requests.get", return_value=_resp(400, body)):
        with pytest.raises(ExternalCallError) as exc:
            request_with_retry("get", "http://x", source="pubchem")
    assert "HTTP 400: Invalid name" in str(exc.value)


@pytest.mark.parametrize("body", ["<html><body>502 Bad Gateway</body></html>", "", "{}", '{"error": {}}'])
def test_html_and_empty_bodies_are_not_quoted(body, sleeps):
    with patch("requests.get", return_value=_resp(400, body)):
        with pytest.raises(ExternalCallError) as exc:
            request_with_retry("get", "http://x", source="s")
    assert "(HTTP 400)" in str(exc.value)


def test_a_long_explanation_is_truncated(sleeps):
    body = json.dumps({"error": {"message": "x" * 1000}})
    with patch("requests.get", return_value=_resp(400, body)):
        with pytest.raises(ExternalCallError) as exc:
            request_with_retry("get", "http://x", source="s")
    assert len(str(exc.value)) < 300 and "…" in str(exc.value)


@pytest.mark.parametrize("leak", [
    "Incorrect API key provided: sk-proj-abc***xyz1. You can find your API key",
    "bad credentials api_key=abcdef123456 rejected",
    "Authorization: Bearer abcdef123456",
])
def test_key_shaped_text_is_scrubbed_from_errors_logs_and_state(leak, sleeps, capsys):
    body = json.dumps({"error": {"message": leak}})
    with patch("requests.get", side_effect=[_resp(503, body), _resp(401, body)]):
        with pytest.raises(ExternalCallError) as exc:
            request_with_retry("get", "http://x", source="s")
    shown = str(exc.value) + capsys.readouterr().out + compact_error(leak)
    assert "[redacted]" in shown
    for secret in ("sk-proj-abc", "abcdef123456"):
        assert secret not in shown


def test_ordinary_words_that_contain_sk_are_left_alone():
    assert compact_error("task-based risk-averse desk-top") == "task-based risk-averse desk-top"

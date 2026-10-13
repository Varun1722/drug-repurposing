"""resilience.py against a real local HTTP server (no mocks of ``requests``).

The mocked tests prove the logic; these prove it holds for real transports:
header parsing, read timeouts, refused connections and real response bodies.
Delays are tiny so the file runs in about a second.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import llm
from resilience import ExternalCallError, RetryPolicy, request_with_retry

FAST = RetryPolicy(max_attempts=4, base_delay=0.01, max_delay=0.02, max_total_seconds=30)


class _Server:
    """Serves a script of ``(status, headers, body, delay_seconds)`` replies, one
    per request (the last repeats), and records what it received."""

    def __init__(self, script):
        self.script = list(script)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _serve(self):
                length = int(self.headers.get("Content-Length") or 0)
                outer.requests.append({
                    "path": self.path,
                    "headers": dict(self.headers),
                    "body": self.rfile.read(length).decode() if length else "",
                })
                status, headers, body, delay = (
                    outer.script.pop(0) if len(outer.script) > 1 else outer.script[0])
                time.sleep(delay)
                payload = body if isinstance(body, bytes) else json.dumps(body).encode()
                try:
                    self.send_response(status)
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client gave up first (timeout test)

            do_GET = do_POST = _serve

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01},
                         daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    made = []

    def make(*script):
        s = _Server(script)
        made.append(s)
        return s

    yield make
    for s in made:
        s.close()


def reply(status, body=None, headers=None, delay=0.0):
    return (status, headers or {}, {} if body is None else body, delay)


def test_success_over_a_real_socket(server):
    s = server(reply(200, {"ok": True}))
    resp = request_with_retry("get", s.url + "/x", source="svc", policy=FAST)
    assert resp.status_code == 200 and resp.json() == {"ok": True}


def test_retry_after_ms_header_is_parsed_from_a_real_response(server):
    s = server(reply(429, headers={"retry-after-ms": "150"}), reply(200, {"ok": 1}))
    started = time.monotonic()
    resp = request_with_retry("get", s.url, source="svc", policy=FAST)
    assert resp.status_code == 200 and len(s.requests) == 2
    assert time.monotonic() - started >= 0.14        # waited what the server asked


def test_retry_after_seconds_header_is_parsed_from_a_real_response(server):
    s = server(reply(503, headers={"Retry-After": "1"}), reply(200, {"ok": 1}))
    started = time.monotonic()
    request_with_retry("get", s.url, source="svc", policy=FAST)
    assert time.monotonic() - started >= 0.95


def test_a_persistent_503_exhausts_the_attempts(server):
    s = server(reply(503))
    with pytest.raises(ExternalCallError) as exc:
        request_with_retry("post", s.url, source="svc", json={"a": 1}, policy=FAST)
    assert (exc.value.kind, exc.value.status, exc.value.attempts) == ("server", 503, 4)
    assert len(s.requests) == 4
    assert all(json.loads(r["body"]) == {"a": 1} for r in s.requests)   # body resent each time


def test_404_is_a_result_when_allowed_and_an_error_when_not(server):
    s = server(reply(404))
    assert request_with_retry("get", s.url, source="svc", allow_status=(404,), policy=FAST).status_code == 404
    with pytest.raises(ExternalCallError) as exc:
        request_with_retry("get", s.url, source="svc", policy=FAST)
    assert exc.value.kind == "client" and len(s.requests) == 2    # one try each, no retries


def test_a_refused_connection_is_retried_then_reported():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]               # nothing listens here once closed
    with pytest.raises(ExternalCallError) as exc:
        request_with_retry("get", f"http://127.0.0.1:{port}", source="svc", timeout=1, policy=FAST)
    assert (exc.value.kind, exc.value.attempts) == ("connection", 4)


def test_a_read_timeout_is_retried_and_can_recover(server):
    s = server(reply(200, {"slow": True}, delay=0.6), reply(200, {"fast": True}))
    resp = request_with_retry("get", s.url, source="svc", timeout=0.2, policy=FAST)
    assert resp.json() == {"fast": True} and len(s.requests) == 2


def test_a_read_timeout_that_never_recovers_is_a_timeout_error(server):
    s = server(reply(200, {}, delay=0.5))
    with pytest.raises(ExternalCallError) as exc:
        request_with_retry("get", s.url, source="svc", timeout=0.1,
                           policy=RetryPolicy(max_attempts=2, base_delay=0.01))
    assert exc.value.kind == "timeout" and exc.value.attempts == 2


# ---------------------------------------------------------------------------
# The OpenAI client wired to the same transport
# ---------------------------------------------------------------------------

def _completion(content):
    return {"choices": [{"finish_reason": "stop", "message": {"content": content}}]}


@pytest.fixture
def openai_at(server, monkeypatch):
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setattr(llm, "OPENAI_POLICY", FAST)

    def point(*script):
        s = server(*script)
        monkeypatch.setattr(llm, "OPENAI_URL", s.url + "/v1/chat/completions")
        return s

    return point


def test_openai_call_sends_auth_and_json_and_returns_the_text(openai_at):
    s = openai_at(reply(200, _completion("hello")))
    assert llm.query_openai("sys", "usr") == "hello"
    sent = s.requests[0]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer sk-test-key"
    assert json.loads(sent["body"])["messages"][1] == {"role": "user", "content": "usr"}


def test_openai_rate_limit_then_success(openai_at):
    s = openai_at(reply(429, {"error": {"code": "rate_limit_exceeded"}},
                        headers={"retry-after-ms": "20"}),
                  reply(200, _completion("recovered")))
    assert llm.query_openai("sys", "usr") == "recovered"
    assert len(s.requests) == 2


def test_openai_exhausted_credit_is_reported_at_once(openai_at):
    body = {"error": {"code": "credit_balance_exhausted", "type": "insufficient_quota",
                      "message": "You exceeded your current quota"}}
    s = openai_at(reply(429, body))
    with pytest.raises(ExternalCallError) as exc:
        llm.query_openai("sys", "usr")
    assert (exc.value.source, exc.value.kind) == ("openai", "quota")
    assert len(s.requests) == 1


def test_openai_bad_key_is_an_auth_error_without_retries(openai_at):
    s = openai_at(reply(401, {"error": {"message": "Incorrect API key provided"}}))
    with pytest.raises(ExternalCallError) as exc:
        llm.query_openai("sys", "usr")
    assert exc.value.kind == "auth" and len(s.requests) == 1


def test_openai_malformed_success_body_is_a_bad_response(openai_at):
    openai_at(reply(200, b"<html>gateway</html>"))
    with pytest.raises(ExternalCallError) as exc:
        llm.query_openai("sys", "usr")
    assert exc.value.kind == "bad_response"

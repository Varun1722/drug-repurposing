"""Tests for Agent 3's docking client: chunk retries and failure reporting.

The GNINA worker is mocked at ``requests.post``; no network, no real sleeping.
"""

import json

import pytest
import requests

import agent3
from resilience import DOCKING_POLICY, ExternalCallError


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body if body is not None else {"results": []}
        self.text = text
        self.headers = {}

    def json(self):
        return self._body


def _result(name, score=0.5):
    return {"name": name, "confidence_score": score, "confidence_raw": score}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)


@pytest.fixture
def pdb(tmp_path):
    path = tmp_path / "x.pdb"
    path.write_text("HEADER\nEND\n")
    return str(path)


def _ligands(n):
    return [{"name": f"lig{i}", "smiles": "C" * (i + 1)} for i in range(n)]


def _by_chunk_ligands(payload):
    return [lig["name"] for lig in payload["ligands"]]


# ---------------------------------------------------------------------------
# submit_chunk
# ---------------------------------------------------------------------------

def test_transient_worker_error_is_retried(monkeypatch):
    replies = [_Resp(503), _Resp(200, {"results": [_result("a")]})]
    monkeypatch.setattr(requests, "post", lambda *a, **k: replies.pop(0))
    out = agent3.submit_chunk("http://w", "b64", _ligands(1), 10, 0)
    assert out["results"][0]["name"] == "a" and replies == []


def test_chunk_is_retried_per_the_docking_policy_with_a_long_timeout(monkeypatch):
    seen = []

    def post(url, json=None, timeout=None):  # noqa: A002
        seen.append(timeout)
        return _Resp(500)

    monkeypatch.setattr(requests, "post", post)
    with pytest.raises(ExternalCallError) as exc:
        agent3.submit_chunk("http://w", "b64", _ligands(1), 10, 0)
    assert exc.value.source == "gnina"
    assert seen == [600] * DOCKING_POLICY.max_attempts


def test_bad_payload_is_not_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or _Resp(422))
    with pytest.raises(ExternalCallError) as exc:
        agent3.submit_chunk("http://w", "b64", _ligands(1), 10, 0)
    assert exc.value.kind == "client" and len(calls) == 1


# ---------------------------------------------------------------------------
# run_docking
# ---------------------------------------------------------------------------

def test_all_chunks_succeed(monkeypatch, pdb):
    def post(url, json=None, timeout=None):  # noqa: A002
        return _Resp(200, {"results": [_result(n, 0.1 * (i + 1))
                                       for i, n in enumerate(_by_chunk_ligands(json))]})

    monkeypatch.setattr(requests, "post", post)
    errors = []
    results, elapsed = agent3.run_docking(pdb, _ligands(5), endpoint_url="http://w",
                                          chunk_size=2, errors=errors)
    assert len(results) == 5 and errors == []
    assert results == sorted(results, key=lambda r: r["confidence_score"], reverse=True)


def test_one_failed_chunk_is_reported_and_the_rest_are_kept(monkeypatch, pdb):
    def post(url, json=None, timeout=None):  # noqa: A002
        names = _by_chunk_ligands(json)
        if "lig2" in names:                       # the middle chunk always fails
            return _Resp(503)
        return _Resp(200, {"results": [_result(n) for n in names]})

    monkeypatch.setattr(requests, "post", post)
    errors = []
    results, _ = agent3.run_docking(pdb, _ligands(6), endpoint_url="http://w",
                                    chunk_size=2, errors=errors)

    assert sorted(r["name"] for r in results) == ["lig0", "lig1", "lig4", "lig5"]
    assert len(errors) == 1
    assert errors[0]["kind"] == "server" and errors[0]["chunk"] == 1 and errors[0]["count"] == 2
    assert "gnina" in errors[0]["message"]


def test_every_chunk_failing_raises_instead_of_returning_nothing(monkeypatch, pdb):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(502))
    errors = []
    with pytest.raises(ExternalCallError) as exc:
        agent3.run_docking(pdb, _ligands(4), endpoint_url="http://w", chunk_size=2,
                           errors=errors)
    assert exc.value.source == "gnina" and "all 2 docking chunk(s) failed" in str(exc.value)
    assert len(errors) == 2


def test_worker_reported_ligand_failures_are_collected(monkeypatch, pdb):
    body = {"results": [_result("lig0")],
            "errors": [{"name": "lig1", "error": "bad SMILES"},
                       {"name": "lig2", "error": "embed failed"}]}
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200, body))
    errors = []
    results, _ = agent3.run_docking(pdb, _ligands(3), endpoint_url="http://w", errors=errors)

    assert [r["name"] for r in results] == ["lig0"]
    assert errors[0]["kind"] == "ligand_failed" and errors[0]["count"] == 2
    assert "lig1: bad SMILES" in errors[0]["message"]


def test_a_bug_in_a_chunk_is_recorded_as_unexpected_not_swallowed(monkeypatch, pdb):
    def post(url, json=None, timeout=None):  # noqa: A002
        if "lig0" in _by_chunk_ligands(json):
            raise KeyError("boom")
        return _Resp(200, {"results": [_result("lig1")]})

    monkeypatch.setattr(requests, "post", post)
    errors = []
    results, _ = agent3.run_docking(pdb, _ligands(2), endpoint_url="http://w",
                                    chunk_size=1, errors=errors)
    assert [r["name"] for r in results] == ["lig1"]
    assert errors[0]["kind"] == "unexpected" and "KeyError" in errors[0]["message"]


def test_no_ligands_is_not_an_error(pdb):
    assert agent3.run_docking(pdb, [], endpoint_url="http://w") == ([], 0.0)


def test_missing_endpoint_is_a_configuration_error(pdb, monkeypatch):
    monkeypatch.delenv("GNINA_ENDPOINT_URL", raising=False)
    with pytest.raises(ValueError, match="GNINA_ENDPOINT_URL"):
        agent3.run_docking(pdb, _ligands(1))


# ---------------------------------------------------------------------------
# run_pipeline (CLI pipeline mode)
# ---------------------------------------------------------------------------

def test_pipeline_mode_marks_a_target_error_and_keeps_going(monkeypatch, tmp_path, pdb):
    agent2_out = tmp_path / "in.json"
    agent2_out.write_text(json.dumps({"disease": "d", "targets": [
        {"protein": "BAD", "pdb_id": "1", "pdb_file": pdb, "ligands": _ligands(1)},
        {"protein": "GOOD", "pdb_id": "2", "pdb_file": pdb,
         "ligands": [{"name": "ok", "smiles": "CC"}]},
    ]}))

    def post(url, json=None, timeout=None):  # noqa: A002
        if json["ligands"][0]["name"] == "lig0":
            return _Resp(503)
        return _Resp(200, {"results": [_result("ok")]})

    monkeypatch.setattr(requests, "post", post)
    out_path = tmp_path / "out.json"
    result = agent3.run_pipeline(str(agent2_out), str(out_path), endpoint_url="http://w")

    bad, good = result["targets"]
    assert bad["status"] == "error" and "gnina" in bad["error"]
    assert good["status"] == "completed" and good["num_ligands_docked"] == 1
    assert result["status"] == "completed"
    assert json.loads(out_path.read_text())["targets"][0]["status"] == "error"

"""Tests for schema-constrained OpenAI calls (llm_schemas.py + llm.py + the review.py call sites).

HTTP is mocked, so these need no API key and make no network calls.
"""

import json
from typing import List, Optional

import pytest
import requests
from pydantic import BaseModel, ConfigDict

import llm
import review
from resilience import ExternalCallError
from llm_schemas import (
    DrugNameList,
    DrugProteinMap,
    ProteinList,
    ReasoningDecision,
    to_strict_json_schema,
)

ALL_MODELS = [ProteinList, DrugNameList, DrugProteinMap, ReasoningDecision]


class _Strict(BaseModel):
    """Base for throwaway test models: closed objects, so each test isolates
    the one rule it is about."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model", ALL_MODELS, ids=lambda m: m.__name__)
def test_shipped_models_are_strict_safe(model):
    schema = to_strict_json_schema(model)
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert "default" not in json.dumps(schema)


def test_decision_schema_puts_reasoning_before_action_and_requires_all_keys():
    schema = to_strict_json_schema(ReasoningDecision)
    keys = list(schema["properties"])
    assert keys[:3] == ["rationale", "hypothesis", "action"]
    assert schema["required"] == keys


def test_nested_models_are_strict_too():
    schema = to_strict_json_schema(DrugProteinMap)
    entry = schema["$defs"]["DrugEntry"]
    assert entry["additionalProperties"] is False
    assert set(entry["required"]) == set(entry["properties"])


def test_guard_rejects_default_values():
    # A field with a default is left out of `required` -- strict mode needs
    # every key required, and the message says how to express "nullable".
    class WithDefault(_Strict):
        a: str = "x"

    with pytest.raises(ValueError, match=r"non-required properties \(a\).*Optional"):
        to_strict_json_schema(WithDefault)


def test_guard_rejects_optional_with_default_none():
    class OptionalWithDefault(_Strict):
        a: Optional[str] = None

    with pytest.raises(ValueError, match=r"non-required properties \(a\)"):
        to_strict_json_schema(OptionalWithDefault)


def test_guard_rejects_extra_allow():
    class Loose(BaseModel):
        model_config = ConfigDict(extra="allow")
        a: str

    with pytest.raises(ValueError, match="additionalProperties: false"):
        to_strict_json_schema(Loose)


def test_guard_rejects_unwrapped_list_root():
    with pytest.raises(Exception):
        # not a BaseModel with an object schema
        to_strict_json_schema(List[str])  # type: ignore[arg-type]


def test_field_named_like_a_keyword_is_allowed():
    class Keywordish(_Strict):
        default: str
        not_: str

    to_strict_json_schema(Keywordish)


# ---------------------------------------------------------------------------
# query_openai_structured
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, choice, status=200, text="", headers=None):
        self._choice = choice
        self.status_code = status
        self.ok = status == 200
        self.text = text
        self.headers = headers or {}

    def json(self):
        return {"choices": [self._choice]}


@pytest.fixture
def post(monkeypatch):
    """Patch requests.post (and sleeping, so retries are instant).

    Returns ``reply(choice, status=200, text="", headers=None)``, which sets the
    reply used for every request, and ``reply.script([...])``, which queues
    one ``(choice, status)`` per request before falling back to that reply.
    ``reply.sent`` captures request payloads and ``reply.sleeps`` the backoff
    waits."""
    monkeypatch.setattr(llm, "OPENAI_API_KEY", "test-key")
    sleeps = []
    monkeypatch.setattr("time.sleep", sleeps.append)
    sent = []
    queue = []
    state = {"choice": None, "status": 200, "text": "", "headers": None}

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        sent.append(json)
        if queue:
            choice, status = queue.pop(0)
            return _FakeResponse(choice, status)
        return _FakeResponse(state["choice"], state["status"],
                             state["text"], state["headers"])

    monkeypatch.setattr(requests, "post", fake_post)

    def reply(choice, status=200, text="", headers=None):
        state.update(choice=choice, status=status, text=text, headers=headers)

    reply.script = lambda items: queue.extend(items)
    reply.sent = sent
    reply.sleeps = sleeps
    return reply


def _ok(content, finish_reason="stop"):
    return {"finish_reason": finish_reason, "message": {"content": content}}


def _kind(post, choice, model=ProteinList):
    post(choice)
    with pytest.raises(llm.LLMOutputError) as exc:
        llm.query_openai_structured("sys", "usr", model)
    return exc.value.kind


def test_success_returns_validated_model_and_sends_strict_json_schema(post):
    post(_ok(json.dumps({"proteins": ["EGFR", "BRAF"]})))
    result = llm.query_openai_structured("sys", "usr", ProteinList)

    assert isinstance(result, ProteinList)
    assert result.proteins == ["EGFR", "BRAF"]

    payload = post.sent[0]
    fmt = payload["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["name"] == "ProteinList"
    assert fmt["json_schema"]["schema"]["additionalProperties"] is False
    assert payload["max_completion_tokens"] == 8000
    assert "temperature" not in payload


def test_refusal(post):
    choice = {"finish_reason": "stop", "message": {"content": None, "refusal": "no"}}
    assert _kind(post, choice) == "refusal"


def test_truncated(post):
    assert _kind(post, _ok('{"proteins": ["EG', finish_reason="length")) == "truncated"


def test_content_filter(post):
    assert _kind(post, _ok("{", finish_reason="content_filter")) == "content_filter"


def test_empty_content(post):
    assert _kind(post, _ok("")) == "empty"


def test_invalid_json(post):
    assert _kind(post, _ok("not json")) == "invalid_json"


def test_wrong_type_is_schema_mismatch(post):
    assert _kind(post, _ok('{"proteins": "EGFR"}')) == "schema_mismatch"


def test_invalid_enum_is_schema_mismatch(post):
    bad = {
        "rationale": "r", "hypothesis": "h", "action": "delete_everything",
        "seed_cids": None, "drug_class": None, "drug_names": None,
    }
    assert _kind(post, _ok(json.dumps(bad)), ReasoningDecision) == "schema_mismatch"


def test_decision_with_nulls_round_trips(post):
    post(_ok(json.dumps({
        "rationale": "r", "hypothesis": "h", "action": "expand_class",
        "seed_cids": None, "drug_class": "statins", "drug_names": ["atorvastatin"],
    })))
    decision = llm.query_openai_structured("s", "u", ReasoningDecision)
    assert decision.action == "expand_class"
    assert decision.seed_cids is None
    assert decision.drug_names == ["atorvastatin"]


# ---------------------------------------------------------------------------
# Retries: transport failures
# ---------------------------------------------------------------------------

def test_transient_http_errors_are_retried_then_succeed(post):
    good = _ok(json.dumps({"proteins": ["EGFR"]}))
    post.script([({}, 503), ({}, 429)])
    post(good)
    result = llm.query_openai_structured("sys", "usr", ProteinList)

    assert result.proteins == ["EGFR"]
    assert len(post.sent) == 3
    assert len(post.sleeps) == 2


def test_persistent_server_error_raises_external_call_error(post):
    post({}, status=503)
    with pytest.raises(ExternalCallError) as exc:
        llm.query_openai_structured("sys", "usr", ProteinList)

    assert exc.value.source == "openai"
    assert exc.value.kind == "server"
    assert exc.value.attempts == llm.OPENAI_POLICY.max_attempts
    assert len(post.sent) == llm.OPENAI_POLICY.max_attempts


def test_rate_limit_honours_retry_after(post):
    post({}, status=429, headers={"retry-after": "7"})
    with pytest.raises(ExternalCallError) as exc:
        llm.query_openai_structured("sys", "usr", ProteinList)

    assert exc.value.kind == "rate_limited"
    assert exc.value.attempts == llm.OPENAI_POLICY.max_attempts
    assert post.sleeps and all(wait >= 7 for wait in post.sleeps)


def test_retry_after_longer_than_the_cap_is_surfaced_not_slept(post):
    post({}, status=429, headers={"retry-after": "600"})
    with pytest.raises(ExternalCallError) as exc:
        llm.query_openai_structured("sys", "usr", ProteinList)

    assert exc.value.kind == "rate_limited"
    assert post.sleeps == [] and len(post.sent) == 1


def test_exhausted_credit_is_not_retried(post):
    body = json.dumps({"error": {"code": "credit_balance_exhausted",
                                 "type": "insufficient_quota"}})
    post({}, status=429, text=body)
    with pytest.raises(ExternalCallError) as exc:
        llm.query_openai_structured("sys", "usr", ProteinList)

    assert exc.value.kind == "quota"
    assert len(post.sent) == 1 and post.sleeps == []


def test_bad_request_is_not_retried(post):
    post({}, status=400)
    with pytest.raises(ExternalCallError) as exc:
        llm.query_openai("sys", "usr")

    assert exc.value.kind == "client"
    assert len(post.sent) == 1


def test_response_without_choices_is_a_bad_response(post, monkeypatch):
    class NoChoices(_FakeResponse):
        def json(self):
            return {"error": "?"}

    monkeypatch.setattr(requests, "post", lambda *a, **k: NoChoices(None))
    with pytest.raises(ExternalCallError) as exc:
        llm.query_openai("sys", "usr")

    assert exc.value.kind == "bad_response"


# ---------------------------------------------------------------------------
# Retries: unusable replies ("repair" attempt)
# ---------------------------------------------------------------------------

def test_malformed_reply_is_retried_with_the_problem_in_the_prompt(post):
    good = _ok(json.dumps({"proteins": ["EGFR"]}))
    post.script([(_ok('{"proteins": "EGFR"}'), 200)])
    post(good)
    result = llm.query_openai_structured("sys", "original prompt", ProteinList)

    assert result.proteins == ["EGFR"]
    assert len(post.sent) == 2
    first_user = post.sent[0]["messages"][1]["content"]
    second_user = post.sent[1]["messages"][1]["content"]
    assert first_user == "original prompt"
    assert second_user.startswith("original prompt")
    assert "previous reply was rejected" in second_user
    assert "failed validation" in second_user


def test_still_malformed_after_second_try_raises_with_attempt_count(post):
    post(_ok("not json"))
    with pytest.raises(llm.LLMOutputError) as exc:
        llm.query_openai_structured("sys", "usr", ProteinList)

    assert exc.value.kind == "invalid_json"
    assert exc.value.attempts == 2
    assert len(post.sent) == 2


@pytest.mark.parametrize("choice", [
    {"finish_reason": "stop", "message": {"content": None, "refusal": "no"}},
    _ok('{"proteins": ["EG', finish_reason="length"),
    _ok("{", finish_reason="content_filter"),
], ids=["refusal", "truncated", "content_filter"])
def test_refusal_truncation_and_filter_are_not_retried(post, choice):
    post(choice)
    with pytest.raises(llm.LLMOutputError) as exc:
        llm.query_openai_structured("sys", "usr", ProteinList)

    assert exc.value.attempts == 1
    assert len(post.sent) == 1


def test_plain_query_openai_is_unchanged(post):
    post({"finish_reason": "stop", "message": {"content": "hello"}})
    assert llm.query_openai("sys", "usr") == "hello"
    assert "response_format" not in post.sent[0]


# ---------------------------------------------------------------------------
# Call-site fallbacks
# ---------------------------------------------------------------------------

def test_protein_extraction_dedupes_and_trims(post):
    post(_ok(json.dumps({"proteins": [" EGFR ", "EGFR", "", "BRAF"]})))
    assert review.extract_proteins_from_text("text") == ["EGFR", "BRAF"]


def test_protein_extraction_returns_empty_on_refusal(post):
    post({"finish_reason": "stop", "message": {"content": None, "refusal": "no"}})
    # [] makes the pipeline use extract_proteins_fallback -- no junk tokens.
    assert review.extract_proteins_from_text("text") == []


def test_prose_reply_no_longer_becomes_protein_names(post):
    post(_ok("Sure! Here are some proteins, namely EGFR, BRAF and KRAS."))
    assert review.extract_proteins_from_text("text") == []


def test_drug_map_success(post):
    post(_ok(json.dumps({"drugs": [{
        "drug": "sotorasib", "proteins": ["KRAS"], "mechanism": "covalent inhibitor",
        "fda_status": "Approved", "category": "mainstream",
        "trial_status_for_targets": "Approved for KRAS G12C NSCLC",
    }]})))
    data = review._extract_drug_protein_map("review text", "NSCLC", ["KRAS"])
    assert data["drugs"][0]["drug"] == "sotorasib"
    assert data["disease"] == "NSCLC"
    assert data["protein_targets"] == ["KRAS"]
    assert "llm_error" not in data


def test_drug_map_failure_is_recorded_not_silent(post):
    post(_ok('{"drugs": [', finish_reason="length"))
    data = review._extract_drug_protein_map("review text", "NSCLC", ["KRAS"])
    assert data["drugs"] == []
    assert data["llm_error"]["kind"] == "truncated"


def test_analyze_and_decide_falls_back_to_proceed_and_records_why(post):
    pytest.importorskip("rdkit")  # pipeline imports agent2, which needs RDKit
    import pipeline

    post({"finish_reason": "stop", "message": {"content": None, "refusal": "no"}})
    state = {
        "disease": "NSCLC", "round": 1, "all_docking_results": [],
        "hypotheses": [], "expansion_history": [],
    }
    # analyze_and_decide persists state; keep it out of the repo
    pipeline.save_state = lambda s: None
    decision = pipeline.analyze_and_decide(state)

    assert decision["action"] == "proceed"
    assert "refusal" in decision["rationale"]
    assert state["expansion_history"][0]["action"] == "proceed"
    # ... and is recorded as a (non-fatal) error in the state, not just the rationale
    assert [(e["stage"], e["source"], e["kind"], e["fatal"]) for e in state["errors"]] == [
        ("reasoning", "openai", "refusal", False)
    ]

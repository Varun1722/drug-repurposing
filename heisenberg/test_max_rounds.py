"""--max-rounds: one limit, honoured by the loop, the cutoff and the prompt.

The real run_pipeline loop runs with every stage and external call faked and a
model that is scripted per test.
"""

import json

import pytest

pytest.importorskip("rdkit")  # pipeline imports agent2, which needs RDKit

import pipeline  # noqa: E402
from llm_schemas import ReasoningDecision  # noqa: E402
from resilience import ExternalCallError  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("time.sleep", lambda _: None)


def _decision(action="expand_class", names=("a",)):
    return ReasoningDecision(
        rationale="r", hypothesis="h", action=action, seed_cids=None,
        drug_class="statins" if action == "expand_class" else None,
        drug_names=list(names) if action == "expand_class" else None)


class Run:
    """Drives run_pipeline with a scripted model; records what the model was asked."""

    def __init__(self, monkeypatch, decisions=None, finds_ligands=True):
        self.asked = []                       # (round, max_round) per reasoning call
        decisions = list(decisions or [])

        def llm(prompt_id, model, **v):
            if prompt_id != "analysis/reasoning_decision":
                raise AssertionError(prompt_id)
            self.asked.append((v["round"], v["max_round"]))
            return decisions.pop(0) if decisions else _decision()

        def docking(state, targets_to_dock=None):
            state["all_docking_results"].append({
                "name": "x", "confidence_score": .5, "mechanism": "", "fda_status": "",
                "source": "", "protein_target": "KRAS", "round": state["round"]})
            return state

        def structure(state):
            state["targets"] = [{"protein": "KRAS", "pdb_id": "1", "pdb_file": "f.pdb", "ligands": []}]
            return state

        monkeypatch.setattr(pipeline, "query_prompt_structured", llm)
        monkeypatch.setattr(pipeline, "lookup_smiles",
                            lambda n: {"cid": 1, "smiles": "C"} if finds_ligands else None)
        monkeypatch.setattr(pipeline, "stage_docking", docking)
        monkeypatch.setattr(pipeline, "stage_structure", structure)
        monkeypatch.setattr(pipeline, "stage_literature",
                            lambda s: s.update(protein_targets=["KRAS"]) or s)
        monkeypatch.setattr(pipeline, "stage_report", lambda s: s)
        monkeypatch.setattr(pipeline, "stage_paper", lambda s: s)

    def go(self, max_rounds):
        self.state = pipeline.run_pipeline("d", max_rounds=max_rounds)
        history = self.state["expansion_history"]
        self.expansions = sum(1 for h in history if h["action"] != "proceed")
        return self


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5])
def test_the_flag_is_the_limit_for_any_value(monkeypatch, n):
    run = Run(monkeypatch).go(n)          # a model that always wants to expand

    assert len(run.asked) == n            # one question per allowed round, none wasted
    assert run.expansions == n
    assert run.state["round"] == n + 1    # round 1 is the initial docking
    assert not any("Max rounds" in h["rationale"] for h in run.state["expansion_history"])


def test_the_model_is_told_the_real_limit(monkeypatch):
    run = Run(monkeypatch).go(3)
    assert run.asked == [(1, 4), (2, 4), (3, 4)]


def test_the_default_is_still_two(monkeypatch):
    run = Run(monkeypatch)
    pipeline.run_pipeline("d")
    assert [r for r, _ in run.asked] == [1, 2]
    assert pipeline.new_state("d")["max_rounds"] == 2


def test_proceed_ends_the_loop_early(monkeypatch):
    run = Run(monkeypatch, decisions=[_decision(), _decision("proceed")]).go(5)
    assert len(run.asked) == 2 and run.expansions == 1 and run.state["round"] == 2


def test_an_expansion_that_finds_nothing_still_uses_up_a_pass(monkeypatch):
    run = Run(monkeypatch, finds_ligands=False).go(3)
    assert len(run.asked) == 3            # bounded, even though the round never advanced
    assert run.state["round"] == 1


def test_the_limit_is_saved_in_the_state_file(monkeypatch):
    Run(monkeypatch).go(4)
    assert json.load(open(pipeline.STATE_FILE))["max_rounds"] == 4


def test_a_negative_limit_is_rejected():
    with pytest.raises(ValueError, match="max_rounds"):
        pipeline.run_pipeline("d", max_rounds=-1)


def test_the_cli_rejects_a_negative_flag(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["pipeline.py", "d", "--max-rounds", "-2"])
    with pytest.raises(SystemExit) as exc:
        pipeline.main()
    assert exc.value.code == 2 and "must be >= 0" in capsys.readouterr().err


def test_the_cli_passes_the_flag_through(monkeypatch):
    seen = {}
    monkeypatch.setattr("sys.argv", ["pipeline.py", "d", "--max-rounds", "4"])
    monkeypatch.setattr(pipeline, "run_pipeline", lambda d, max_rounds: seen.update(n=max_rounds))
    pipeline.main()
    assert seen == {"n": 4}


# ---------------------------------------------------------------------------
# analyze_and_decide at the limit, and resume
# ---------------------------------------------------------------------------

def _state(max_rounds, round_, history=()):
    state = pipeline.new_state("d", max_rounds)
    state.update(round=round_, expansion_history=list(history),
                 all_docking_results=[{"name": "x", "confidence_score": .5}])
    return state


def test_at_the_limit_the_model_is_not_asked(monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("the model must not be called at the limit")

    monkeypatch.setattr(pipeline, "query_prompt_structured", forbidden)
    state = _state(max_rounds=3, round_=4)
    decision = pipeline.analyze_and_decide(state)

    assert decision["action"] == "proceed" and "Max rounds" in decision["rationale"]
    assert state["expansion_history"] == [] and state["hypotheses"] == []   # nothing to record


def test_below_the_limit_it_uses_the_runs_limit_not_a_constant(monkeypatch):
    asked = []
    monkeypatch.setattr(pipeline, "query_prompt_structured",
                        lambda pid, m, **v: asked.append(v["max_round"]) or _decision("proceed"))
    pipeline.analyze_and_decide(_state(max_rounds=5, round_=4))   # beyond the old constant of 3
    assert asked == [6]


def test_old_state_files_default_to_the_standard_limit(tmp_path):
    (tmp_path / pipeline.STATE_FILE).write_text(json.dumps({"disease": "d", "status": "x"}))
    assert pipeline.load_state()["max_rounds"] == pipeline.MAX_EXPANSION_ROUNDS


@pytest.fixture
def stages(monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, "_write_docking_output", lambda s: None)
    monkeypatch.setattr(pipeline, "stage_report", lambda s: calls.append("report") or s)
    monkeypatch.setattr(pipeline, "stage_paper", lambda s: calls.append("paper") or s)
    monkeypatch.setattr(pipeline, "analyze_and_decide",
                        lambda s: calls.append("decide") or {"action": "proceed"})
    return calls


def test_resume_after_the_last_allowed_expansion_does_not_ask_the_model(stages):
    # Default limit 2: rounds 1 and 2 were reasoned about, round 3 (the last
    # expansion's docking) is never reasoned about, in a normal run either.
    history = [{"round": 1, "action": "expand_class", "rationale": "r"},
               {"round": 2, "action": "expand_class", "rationale": "r"}]
    pipeline.save_state(_state(2, 3, history) | {"status": "docking_round_3_complete"})
    pipeline.resume_pipeline()
    assert stages == ["report", "paper"]


def test_resume_uses_the_limit_saved_with_the_run(stages):
    history = [{"round": 1, "action": "expand_class", "rationale": "r"},
               {"round": 2, "action": "expand_class", "rationale": "r"}]
    pipeline.save_state(_state(5, 3, history) | {"status": "docking_round_3_complete"})
    pipeline.resume_pipeline()
    assert stages == ["decide", "report", "paper"]    # round 3 is within a limit of 5


def test_errors_during_reasoning_do_not_change_the_limit_logic(monkeypatch):
    def down(*a, **k):
        raise ExternalCallError("openai", "server", "openai: server")

    monkeypatch.setattr(pipeline, "query_prompt_structured", down)
    with pytest.raises(ExternalCallError):
        pipeline.analyze_and_decide(_state(2, 1))

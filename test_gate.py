import json

import pytest

import gate

ITEM = {"id": "x", "passage": "The Normans settled in  Normandy in the 10th century.",
        "gold_answers": ["Normandy"], "answerable": True}
NO_ANSWER = {**ITEM, "gold_answers": [], "answerable": False}


def reply(answerable=True, answer="Normandy", quote="The Normans settled in Normandy"):
    return json.dumps({"answerable": answerable, "answer": answer, "quote": quote})


def test_grounded_correct_answer_survives_whitespace_differences():
    assert gate.score_item(ITEM, reply()) == {
        "valid": True, "answered": True, "grounded": True, "correct": True, "unsupported": False}


def test_quote_not_in_passage_is_unsupported():
    result = gate.score_item(ITEM, reply(quote="The Normans lived in Normandy"))
    assert result["correct"] and not result["grounded"] and result["unsupported"]


def test_answer_missing_from_quote_is_not_grounded():
    result = gate.score_item(ITEM, reply(quote="in the 10th century"))
    assert not result["grounded"] and not result["unsupported"]


def test_answer_on_unanswerable_item():
    result = gate.score_item(NO_ANSWER, reply())
    assert result["unsupported"] and not result["correct"]


def test_decline_on_unanswerable_item():
    result = gate.score_item(NO_ANSWER, reply(False, "", ""))
    assert result["correct"] and not result["answered"] and not result["unsupported"]


@pytest.mark.parametrize("raw", [
    "not json",
    '```json\n{"answerable": true, "answer": "Normandy", "quote": "Normandy"}\n```',
    '{"answerable": "yes", "answer": "Normandy", "quote": "Normandy"}',
    '{"answerable": true, "answer": "Normandy"}',
    "[]",
    None,
])
def test_malformed_response(raw):
    result = gate.score_item(ITEM, raw)
    assert not result["valid"] and not result["answered"] and not result["correct"]


def test_token_f1_threshold():
    passage = "The Normans settled in the Normandy region of northern France."
    item = {**ITEM, "passage": passage, "gold_answers": ["Normandy region"]}
    assert gate.score_item(item, reply(answer="Normandy", quote=passage))["correct"]  # F1 0.67
    item["gold_answers"] = ["Normandy region of northern France"]
    assert not gate.score_item(item, reply(answer="Normandy", quote=passage))["correct"]  # F1 0.33


def test_regressions_allow_two_points():
    base = {"accuracy": 80.0, "unsupported_answer_rate": 5.0}
    assert gate.regressions({"accuracy": 78.0, "unsupported_answer_rate": 7.0}, base) == []
    assert len(gate.regressions({"accuracy": 77.9, "unsupported_answer_rate": 7.1}, base)) == 2


def test_check_fails_without_recording(tmp_path, monkeypatch, capsys):
    changed = tmp_path / "extract.txt"
    changed.write_text("a changed prompt")
    monkeypatch.setattr(gate, "PROMPT", changed)
    assert gate.check() == 1
    assert "Record responses for the changed prompt" in capsys.readouterr().out


def test_check_passes_on_committed_recording():
    assert gate.check() == 0


def test_weaker_prompt_fails_the_gate():
    _, responses = gate.load_recording(gate.recording_path(gate.ROOT / "prompts/extract_no_quote_rule.txt"))
    baseline = json.loads(gate.BASELINE.read_text())
    assert gate.regressions(gate.summarize(responses), baseline)

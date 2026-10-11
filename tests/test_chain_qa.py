"""Offline tests for the chain director's QA gate and K0 prompt.

No API key, no network: the vision call is replaced by a fake response.
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("dashscope")

from src import chain, media  # noqa: E402  (after importorskip)


def _rsp(text, status_code=200, code=""):
    msg = SimpleNamespace(content=[{"text": text}])
    out = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    return SimpleNamespace(status_code=status_code, code=code, output=out)


@pytest.fixture()
def fake_qa(monkeypatch):
    """Returns a setter: fake_qa(rsp) or fake_qa(exc) controls what the call yields."""
    monkeypatch.delenv("CHAIN_QA_FAIL_OPEN", raising=False)

    def setup(result):
        def with_retry(fn, attempts=4, wait=6):
            if isinstance(result, Exception):
                raise result
            return result
        monkeypatch.setattr(media, "with_retry", with_retry)
    return setup


def test_pass_verdict(fake_qa):
    fake_qa(_rsp('{"same_character": true, "artifacts": false, "issues": ""}'))
    qa = chain.vision_qa("cand.png", "ref.png", "tail")
    assert qa["passed"] is True and qa["status"] == "ok"


def test_fail_verdict(fake_qa):
    fake_qa(_rsp('```json\n{"same_character": false, "artifacts": false, "issues": "crest"}\n```'))
    qa = chain.vision_qa("cand.png", "ref.png", "tail")
    assert qa["passed"] is False and qa["status"] == "ok" and qa["issues"] == "crest"


def test_network_failure_is_fail_closed(fake_qa):
    fake_qa(ConnectionError("reset"))
    qa = chain.vision_qa("cand.png", "ref.png", "tail")
    assert qa["passed"] is False and qa["status"] == "qa_unavailable"


def test_http_error_is_fail_closed(fake_qa):
    fake_qa(_rsp("", status_code=500, code="InternalError"))
    qa = chain.vision_qa("cand.png", "ref.png", "tail")
    assert qa["passed"] is False and qa["status"] == "qa_unavailable"


@pytest.mark.parametrize("reply", ["looks fine to me", "{not json}", '{"issues": "x"}'])
def test_unparseable_reply_is_fail_closed(fake_qa, reply):
    fake_qa(_rsp(reply))
    qa = chain.vision_qa("cand.png", "ref.png", "tail")
    assert qa["passed"] is False and qa["status"] == "qa_unavailable"


@pytest.mark.parametrize("reply", [
    '{"same_character": "false", "artifacts": false}',   # string, not a JSON boolean
    '{"same_character": true}',                          # artifacts missing
    '{"same_character": true, "artifacts": "false"}',
    '{"same_character": true, "artifacts": null}',
    '{"same_character": 1, "artifacts": 0}',
    '[true, false]',                                     # not a JSON object
])
def test_incomplete_or_mistyped_verdict_is_fail_closed(fake_qa, reply):
    fake_qa(_rsp(reply))
    qa = chain.vision_qa("cand.png", "ref.png", "tail")
    assert qa["passed"] is False and qa["status"] == "qa_unavailable"


def test_fail_open_opt_in(fake_qa, monkeypatch):
    fake_qa(ConnectionError("reset"))
    monkeypatch.setenv("CHAIN_QA_FAIL_OPEN", "1")
    qa = chain.vision_qa("cand.png", "ref.png", "tail")
    assert qa["passed"] is True and qa["status"] == "qa_unavailable"


def test_gate_stops_on_unavailable(fake_qa):
    fake_qa(ConnectionError("reset"))
    assert chain._qa_gate("cand.png", "ref.png", "tail") is False


def test_k0_prompt_uses_plan_style():
    shots = [{"keyframe_prompt": "A fox in snow. Cinematic film still. hand-drawn watercolor."}]
    prompt = chain.k0_prompt({"style": "hand-drawn watercolor"}, shots)
    assert prompt == shots[0]["keyframe_prompt"]
    assert "wildlife" not in prompt.lower()


def test_k0_prompt_neutral_default_without_style():
    shots = [{"keyframe_prompt": "A fox in snow. Cinematic film still. ."}]
    prompt = chain.k0_prompt({}, shots)
    assert prompt.endswith(chain.DEFAULT_STYLE)

import io
import json

from clb.ollama_teacher import chat, teacher_generate, unload


class FakeResp(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *a): return False


def fake_urlopen(reply, seen):
    def f(req, timeout=None):
        seen.append(json.loads(req.data)); return FakeResp(json.dumps(reply).encode())
    return f


def test_chat_request_shape_and_ctx_cap():
    seen = []
    out = chat("hi", temperature=0.7, seed=5, max_tokens=9, num_ctx=99999, urlopen=fake_urlopen({"message": {"content": "yo"}}, seen))
    b = seen[0]
    assert out == "yo" and b["stream"] is False and b["keep_alive"] == "2m"
    assert b["options"] == {"temperature": 0.7, "seed": 5, "num_predict": 9, "num_ctx": 4096}


def test_chat_error_raises():
    import pytest
    with pytest.raises(RuntimeError):
        chat("hi", urlopen=fake_urlopen({"error": "model not found"}, []))


def test_unload_sets_keep_alive_zero():
    seen = []
    unload(urlopen=fake_urlopen({}, seen))
    assert seen[0]["keep_alive"] == 0


def test_n_samples_use_seed_plus_i_and_resume(tmp_path):
    calls = []
    def ask(p, temperature, seed, max_tokens):
        calls.append((p, seed)); return f"{p}-{seed}"
    part = tmp_path / "x.partial.jsonl"
    r = teacher_generate(["a", "b"], 0.7, seed=10, n=2, partial=part, ask=ask, cool=lambda: 0)
    assert r == [["a-10", "a-11"], ["b-10", "b-11"]] and len(calls) == 4
    calls.clear()
    assert teacher_generate(["a", "b"], 0.7, seed=10, n=2, partial=part, ask=ask, cool=lambda: 0) == r
    assert calls == []  # everything came from the partial file


def test_crash_keeps_finished_requests(tmp_path):
    part = tmp_path / "y.jsonl"
    def ask(p, temperature, seed, max_tokens):
        if p == "b": raise OSError("boom")
        return p
    try:
        teacher_generate(["a", "b"], partial=part, ask=ask, cool=lambda: 0)
    except OSError:
        pass
    assert len(open(part).readlines()) == 1

"""CTC correctness tests; real model parity is documented separately."""

import itertools
import sys
import unicodedata
from types import SimpleNamespace

import numpy as np
import pytest

from podcurate_mlx.alignment import (
    MeCabJapaneseUnits,
    QwenAligner,
    TurkishAligner,
    ctc_viterbi,
    normalize_turkish_reference,
)


def collapse(sequence):
    return [token for i, token in enumerate(sequence)
            if token and (i == 0 or sequence[i - 1] != token)]


@pytest.mark.parametrize("tokens", [[1], [1, 2], [1, 1], [1, 2, 1], [2, 2]])
def test_ctc_matches_exhaustive_best_path(tokens):
    # All 3**5 paths are enumerated independently of the trellis implementation.
    rng = np.random.default_rng(104)
    scores = rng.normal(size=(5, 3))
    scores -= np.log(np.exp(scores).sum(axis=1, keepdims=True))
    candidates = [p for p in itertools.product(range(3), repeat=5) if collapse(p) == tokens]
    expected = max(sum(scores[t, p[t]] for t in range(5)) for p in candidates)
    path = ctc_viterbi(scores, tokens)
    actual_tokens = np.array([tokens[index] if index >= 0 else 0 for index in path])
    assert collapse(actual_tokens) == tokens
    actual = scores[np.arange(5), actual_tokens].sum()
    assert actual == pytest.approx(expected)


def test_ctc_adjacent_repeated_letters_need_blank_frame():
    assert ctc_viterbi(np.zeros((2, 2)), [1, 1]) is None
    path = ctc_viterbi(np.log([[0.01, 0.99], [0.99, 0.01], [0.01, 0.99]]), [1, 1])
    assert list(path) == [0, -1, 1]


@pytest.mark.parametrize("scores,tokens", [
    (np.zeros((3, 2)), [0]),
    (np.zeros((3, 2)), [2]),
    (np.full((3, 2), np.nan), [1]),
    (np.zeros(4), [1]),
])
def test_ctc_rejects_invalid_inputs(scores, tokens):
    with pytest.raises(ValueError):
        ctc_viterbi(scores, tokens)


def test_turkish_case_is_locale_correct_and_digits_stay_visible():
    assert normalize_turkish_reference("IŞIK, İSTANBUL'da 2026!") == "ışık istanbulda 2026"
    assert normalize_turkish_reference("  Çığ\n ÖĞÜT  ") == "çığ öğüt"


def fake_turkish():
    model = TurkishAligner.__new__(TurkishAligner)
    model._model = object()
    model._receptive, model._stride, model._blank = 400, 320, 0
    model._vocab = {"<pad>": 0, "a": 1, "b": 2, "|": 3}
    return model


def test_turkish_word_spans_follow_ctc_frames_and_score_is_not_probability():
    model = fake_turkish()
    logits = np.full((8, 4), -10.0)
    logits[np.arange(8), [0, 1, 1, 3, 0, 2, 2, 0]] = 10.0
    model.emissions = lambda _: logits
    out = model(np.ones(2960, dtype=np.float32), "a b", "tr")
    assert out["alignment"] == [
        {"text": "a", "start": 0.02, "end": 0.06},
        {"text": "b", "start": 0.10, "end": 0.14},
    ]
    assert out["alignment_status"] == "ok"
    assert out["alignment_coverage"] == 1
    assert out["alignment_score"] <= 0


def test_turkish_oov_is_review_not_silently_deleted():
    model = fake_turkish()
    model.emissions = lambda _: pytest.fail("OOV text should not run the model")
    out = model(np.ones(16000, np.float32), "a 42", "tr")
    assert out["alignment_status"] == "review"
    assert out["alignment_unsupported_characters"] == ["2", "4"]
    assert out["alignment_score"] is None


def test_ctc_reported_score_includes_forced_blank_penalty_for_omission():
    model = fake_turkish()
    logits = np.full((5, 4), -10.0)
    logits[np.arange(5), [0, 1, 3, 2, 0]] = 10.0
    model.emissions = lambda _: logits
    complete = model(np.ones(2000, np.float32), "a b", "tr")
    omitted = model(np.ones(2000, np.float32), "a", "tr")
    assert omitted["alignment_score"] < complete["alignment_score"] - 5
    assert omitted["alignment_token_score"] > omitted["alignment_score"]


def test_turkish_impossible_repeated_target_is_review():
    model = fake_turkish()
    model.emissions = lambda _: np.zeros((2, 4))
    out = model(np.ones(720, np.float32), "aa", "tr")
    assert out["alignment_reasons"] == ["no_ctc_alignment_path"]


def fake_qwen(items, expected):
    model = QwenAligner.__new__(QwenAligner)
    model._model = SimpleNamespace(
        aligner_processor=SimpleNamespace(encode_timestamp=lambda *args: (expected, "")),
        generate=lambda **kwargs: SimpleNamespace(items=items),
    )
    return model


def test_qwen_preserves_missing_confidence():
    item = SimpleNamespace(text="hello", start_time=0.2, end_time=0.8)
    out = fake_qwen([item], ["hello"])(np.ones(16000, np.float32), "hello!", "en")
    assert out["alignment_status"] == "ok"
    assert out["alignment_score"] is None
    assert out["alignment_coverage"] == 1


def test_qwen_no_turkish_inference():
    model = fake_qwen([], [])
    with pytest.raises(ValueError, match="tr is unsupported"):
        model(np.ones(16000, np.float32), "merhaba", "tr")


def test_qwen_invalid_spans_get_review_and_lower_coverage():
    items = [SimpleNamespace(text="a", start_time=0.2, end_time=0.2),
             SimpleNamespace(text="b", start_time=0.7, end_time=1.1)]
    out = fake_qwen(items, ["a", "b"])(np.ones(16000, np.float32), "a b", "en")
    assert out["alignment_status"] == "review"
    assert set(out["alignment_reasons"]) == {
        "invalid_alignment_timestamp", "alignment_outside_audio",
        "reference_units_not_fully_aligned",
    }
    assert out["alignment_coverage"] == 0.5
    assert out["alignment_invalid_units"] == [
        {"text": "a", "start": 0.2, "end": 0.2, "reason": "zero_duration"},
    ]
    assert out["alignment_units_total"] == 2


def test_qwen_nonmonotonic_spans_get_review():
    items = [SimpleNamespace(text="a", start_time=0.1, end_time=0.8),
             SimpleNamespace(text="b", start_time=0.6, end_time=0.9)]
    out = fake_qwen(items, ["a", "b"])(np.ones(16000, np.float32), "a b", "en")
    assert out["alignment_reasons"] == ["nonmonotonic_alignment"]


def test_mecab_japanese_units_use_no_nagisa_and_preserve_lexical_text(monkeypatch):
    pytest.importorskip("fugashi")
    pytest.importorskip("unidic_lite")
    monkeypatch.setitem(sys.modules, "nagisa", None)

    def clean(text):
        return "".join(c for c in text if unicodedata.category(c)[0] in {"L", "N"})

    tokenizer = MeCabJapaneseUnits(clean)
    text = "インターネットで「敵対的環境コース」について検索すると、住所が出ます。"
    units = tokenizer(text)
    assert len(units) > 3
    assert "".join(units) == clean(text)
    assert tokenizer.identity["neural"] is False
    assert tokenizer.identity["coverage_unit"] == "mecab_morphological_units"
